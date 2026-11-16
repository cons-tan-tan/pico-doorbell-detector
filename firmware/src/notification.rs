use crate::decision::Chime;

pub const HOST: &str = "ntfy.constantan.dev";

pub fn request(out: &mut impl core::fmt::Write, topic: &str, chime: Chime) -> core::fmt::Result {
    let body = match chime {
        Chime::Entrance => "エントランスからのインターホンが鳴りました",
        Chime::Door => "玄関からのインターホンが鳴りました",
    };
    write!(
        out,
        "POST {topic} HTTP/1.1\r\nHost: {HOST}\r\nContent-Type: text/plain; charset=utf-8\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    )
}

pub fn status(line: &[u8]) -> Option<u16> {
    if line.len() < 14
        || !line.ends_with(b"\r\n")
        || !(line.starts_with(b"HTTP/1.1 ") || line.starts_with(b"HTTP/1.0 "))
        || !line[9..12].iter().all(u8::is_ascii_digit)
        || line[12] != b' '
    {
        return None;
    }
    Some((line[9] - b'0') as u16 * 100 + (line[10] - b'0') as u16 * 10 + (line[11] - b'0') as u16)
}
