#ifndef SHERPA_H_
#define SHERPA_H_
#include <stdint.h>
#include <stdbool.h>

#include <sherpa-onnx/c-api/c-api.h>

typedef struct {
    const SherpaOnnxOnlineRecognizer *recognizer;
    const SherpaOnnxOnlineStream *stream;
} SherpaHandle;

SherpaHandle sherpa_init(const char* tokens, const char* encoder,
    const char* decoder, const char* joiner);
const char* sherpa_transcribe(SherpaHandle handle, float* samples, int len);
void sherpa_close(SherpaHandle handle);

#endif /* SHERPA_H_ */