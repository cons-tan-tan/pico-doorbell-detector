use std::{env, fs, path::PathBuf, process::Command};

#[path = "build/bundle.rs"]
mod bundle;

fn main() {
    println!("cargo:rerun-if-changed=model");
    println!("cargo:rerun-if-changed=native");
    println!("cargo:rerun-if-changed=memory.x");
    println!("cargo:rerun-if-changed=settings");
    for key in [
        "TFLM_SRC",
        "FLATBUFFERS_INCLUDE",
        "CYW43_FIRMWARE_DIR",
        "DOORBELL_MODE",
        "DOORBELL_MODEL_DIR",
    ] {
        println!("cargo:rerun-if-env-changed={key}");
    }
    let out = PathBuf::from(env::var_os("OUT_DIR").unwrap());
    let sealed = match fs::read("settings/wifi.json") {
        Ok(data) => {
            let settings: serde_json::Value =
                serde_json::from_slice(&data).expect("Invalid encrypted Wi-Fi settings JSON");
            assert!(
                settings["format"] == "doorbell-wifi-sealed-v1",
                "Unknown Wi-Fi settings format"
            );
            let ciphertext = settings["ciphertext"]
                .as_str()
                .expect("Missing Wi-Fi ciphertext");
            assert_eq!(ciphertext.len(), 296, "Invalid Wi-Fi ciphertext length");
            ciphertext
                .as_bytes()
                .as_chunks::<2>()
                .0
                .iter()
                .map(|pair| {
                    let text =
                        std::str::from_utf8(pair).expect("Invalid Wi-Fi ciphertext encoding");
                    u8::from_str_radix(text, 16).expect("Invalid Wi-Fi ciphertext encoding")
                })
                .collect::<Vec<_>>()
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Vec::new(),
        Err(_) => panic!("Cannot read encrypted Wi-Fi settings"),
    };
    fs::write(out.join("wifi.sealed"), sealed).unwrap();
    let model = env::var_os("DOORBELL_MODEL_DIR")
        .map_or_else(|| PathBuf::from("model"), PathBuf::from)
        .canonicalize()
        .unwrap();
    println!("cargo:rerun-if-changed={}", model.display());
    println!("cargo:rustc-env=DOORBELL_MODEL_DIR={}", model.display());
    let manifest: serde_json::Value =
        serde_json::from_slice(&fs::read(model.join("manifest.json")).unwrap()).unwrap();
    fs::write(
        out.join("model_config.rs"),
        bundle::configuration(&model, &manifest),
    )
    .unwrap();

    let target = env::var("TARGET").unwrap();
    let device = env::var("CARGO_CFG_TARGET_OS").unwrap() == "none";
    let arm_flags = [
        "-mcpu=cortex-m33",
        "-mthumb",
        "-mfpu=fpv5-sp-d16",
        "-mfloat-abi=hard",
    ];
    if device {
        assert_eq!(
            target, "thumbv8m.main-none-eabihf",
            "unsupported device target"
        );
        let mode = env::var("DOORBELL_MODE").unwrap_or_else(|_| "dev".into());
        assert!(
            mode == "dev" || mode == "prod",
            "DOORBELL_MODE must be dev or prod"
        );
        println!(
            "cargo:rustc-env=DOORBELL_TOPIC={}",
            if mode == "prod" {
                "/doorbell"
            } else {
                "/debug"
            }
        );
        fs::copy("memory.x", out.join("memory.x")).unwrap();
        println!("cargo:rustc-link-search={}", out.display());
        println!("cargo:rustc-link-arg-bin=doorbell=--nmagic");
        println!("cargo:rustc-link-arg-bin=doorbell=-Tlink.x");
        println!("cargo:rustc-link-arg-bin=doorbell=--wrap=memcpy");
        println!(
            "cargo:rustc-link-arg-bin=doorbell=-Map={}/doorbell.map",
            out.display()
        );
    }

    let source = PathBuf::from(env::var_os("TFLM_SRC").expect("Enter nix develop before building"));
    let mut cpp = cc::Build::new();
    cpp.cpp(true)
        .std("c++17")
        .opt_level(3)
        .warnings(false)
        .include(&source)
        .include(env::var_os("FLATBUFFERS_INCLUDE").unwrap())
        .flag("-fno-exceptions")
        .flag("-fno-rtti")
        .flag("-fno-threadsafe-statics")
        .flag("-fno-unwind-tables")
        .flag("-fno-asynchronous-unwind-tables")
        .define("TFLM_REMOVE_VIRTUAL_DELETE", None)
        .define("TF_LITE_DISABLE_X86_NEON", None)
        .define("FLATBUFFERS_LOCALE_INDEPENDENT", "0")
        .define("FLATBUFFERS_PREFER_PRINTF", "1");
    if env::var("PROFILE").unwrap() == "release" {
        cpp.define("NDEBUG", None);
    }
    let micro = source.join("tensorflow/lite/micro");
    for directory in [
        "",
        "arena_allocator",
        "memory_planner",
        "kernels/internal",
        "kernels/internal/reference",
    ] {
        let mut files: Vec<_> = fs::read_dir(micro.join(directory))
            .unwrap()
            .map(|e| e.unwrap().path())
            .collect();
        files.sort();
        for file in files {
            let name = file.file_name().unwrap().to_string_lossy();
            if file.extension().is_some_and(|e| e == "cc")
                && !name.contains("test")
                && !name.starts_with("mock")
                && name != "debug_log.cc"
                && name != "system_setup.cc"
            {
                cpp.file(file);
            }
        }
    }
    for name in [
        "conv",
        "conv_common",
        "fully_connected",
        "fully_connected_common",
        "reshape",
        "reshape_common",
        "softmax",
        "softmax_common",
        "kernel_util",
    ] {
        cpp.file(micro.join(format!("kernels/{name}.cc")));
    }
    cpp.file("native/tflm.cc");
    if device {
        cpp.flags(arm_flags)
            .cpp_link_stdlib(None)
            .define("DOORBELL_BARE_METAL", None);
    }
    cpp.compile("doorbell_tflm");
    if device {
        for library in ["m", "c", "gcc", "nosys"] {
            let result = Command::new("arm-none-eabi-g++")
                .args(arm_flags)
                .arg(format!("-print-file-name=lib{library}.a"))
                .output()
                .unwrap();
            assert!(result.status.success());
            let path = PathBuf::from(String::from_utf8(result.stdout).unwrap().trim());
            assert!(
                path.is_absolute() && path.is_file(),
                "missing Arm library {library}"
            );
            println!(
                "cargo:rustc-link-search=native={}",
                path.parent().unwrap().display()
            );
            println!("cargo:rustc-link-lib=static={library}");
        }
    }
}
