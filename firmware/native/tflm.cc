#include <cstdarg>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <new>

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"

namespace {
using Interpreter = tflite::micro::MicroInterpreter;
using Resolver = tflite::micro::MicroMutableOpResolver<4>;
alignas(16) uint8_t arena[32 * 1024];
alignas(Interpreter) uint8_t interpreter_storage[sizeof(Interpreter)];
alignas(Resolver) uint8_t resolver_storage[sizeof(Resolver)];
Interpreter* interpreter = nullptr;
}

extern "C" int doorbell_init(const uint8_t* bytes, size_t length,
                            float input_scale, int input_zero,
                            float output_scale, int output_zero) {
  if (!bytes || interpreter || reinterpret_cast<uintptr_t>(bytes) % 16 != 0)
    return 1;
  flatbuffers::Verifier verifier(bytes, length);
  if (!tflite::VerifyModelBuffer(verifier)) return 2;
  const auto* model = tflite::GetModel(bytes);
  if (model->version() != TFLITE_SCHEMA_VERSION) return 2;
  auto* resolver = new (resolver_storage) Resolver();
  if (resolver->AddConv2D() != kTfLiteOk || resolver->AddReshape() != kTfLiteOk ||
      resolver->AddFullyConnected() != kTfLiteOk || resolver->AddSoftmax() != kTfLiteOk)
    return 3;
  auto* candidate = new (interpreter_storage) Interpreter(model, *resolver, arena, sizeof(arena));
  if (candidate->AllocateTensors() != kTfLiteOk) return 4;
  if (candidate->inputs_size() != 1 || candidate->outputs_size() != 1) return 5;
  const auto* input = candidate->input(0);
  const auto* output = candidate->output(0);
  if (input->type != kTfLiteInt8 || output->type != kTfLiteInt8 ||
      input->bytes != 2480 || output->bytes != 3 ||
      input->dims->size != 4 || input->dims->data[0] != 1 ||
      input->dims->data[1] != 124 || input->dims->data[2] != 20 || input->dims->data[3] != 1 ||
      output->dims->size != 2 || output->dims->data[0] != 1 || output->dims->data[1] != 3 ||
      input->params.scale != input_scale || input->params.zero_point != input_zero ||
      output->params.scale != output_scale || output->params.zero_point != output_zero)
    return 5;
  interpreter = candidate;
  return 0;
}

extern "C" int doorbell_invoke(const int8_t* input, size_t length,
                              int8_t* output, size_t output_length) {
  if (!interpreter || !input || !output || length != 2480 || output_length != 3)
    return 1;
  std::memcpy(interpreter->input(0)->data.int8, input, length);
  if (interpreter->Invoke() != kTfLiteOk) return 6;
  std::memcpy(output, interpreter->output(0)->data.int8, output_length);
  return 0;
}

extern "C" size_t doorbell_arena_used() {
  return interpreter ? interpreter->arena_used_bytes() : 0;
}

extern "C" void doorbell_log(const char* message);

extern "C" int DebugVsnprintf(char* buffer, size_t size, const char* format, va_list args) {
  return std::vsnprintf(buffer, size, format, args);
}

extern "C" void DebugLog(const char* format, va_list args) {
  char message[192];
  std::vsnprintf(message, sizeof(message), format, args);
  doorbell_log(message);
}

#ifdef DOORBELL_BARE_METAL
// No C heap is reserved; libc must not grow into the audio buffers or core stacks.
extern "C" void* _sbrk(ptrdiff_t) {
  errno = ENOMEM;
  return reinterpret_cast<void*>(-1);
}
#endif
