#include "sherpa.h"

#include <stdio.h>
#include <string.h>

#define SAMPLE_RATE 16000

SherpaHandle sherpa_init(const char* tokens, const char* encoder,
    const char* decoder, const char* joiner) {
    SherpaHandle handle;
    if (strcmp("", tokens) == 0
      || strcmp("", encoder) == 0
      || strcmp("", decoder) == 0
      || strcmp("", joiner) == 0) {
        return handle;
    }
    SherpaOnnxOnlineRecognizerConfig config;
    memset(&config, 0, sizeof(config));

    config.model_config.debug = 0;
    config.model_config.num_threads = 1;
    config.model_config.provider = "cpu";

    config.decoding_method = "greedy_search";

    config.max_active_paths = 4;

    config.feat_config.sample_rate = 16000;
    config.feat_config.feature_dim = 80;

    config.enable_endpoint = 1;
    config.rule1_min_trailing_silence = 2.4;
    config.rule2_min_trailing_silence = 1.2;
    config.rule3_min_utterance_length = 300;

    config.model_config.tokens = tokens;
    config.model_config.transducer.encoder = encoder;
    config.model_config.transducer.decoder = decoder;
    config.model_config.transducer.joiner = joiner;

    handle.recognizer = CreateOnlineRecognizer(&config);  
    handle.stream = CreateOnlineStream(handle.recognizer);  
    return handle;
}

const char* sherpa_transcribe(SherpaHandle handle,
    float* samples, int len) {
    
    AcceptWaveform(handle.stream, SAMPLE_RATE, samples, len);
    while (IsOnlineStreamReady(handle.recognizer, handle.stream)) {
      DecodeOnlineStream(handle.recognizer, handle.stream);
    }

    const SherpaOnnxOnlineRecognizerResult *r =
        GetOnlineStreamResult(handle.recognizer, handle.stream);
    
    char ret_buf[2048] = {0};
    if (strlen(r->text)) {
      strcat(ret_buf, r->text);
    }
    
    if (IsEndpoint(handle.recognizer, handle.stream)) {
      Reset(handle.recognizer, handle.stream);
    }

    DestroyOnlineRecognizerResult(r);
    
    const char ret[strlen(ret_buf)];
    strcpy(ret, ret_buf);
    return ret;
}

void sherpa_close(SherpaHandle handle) {
    float tail_paddings[4800] = { 0 };
    AcceptWaveform(handle.stream, SAMPLE_RATE, tail_paddings, 4800);
    InputFinished(handle.stream);
    while (IsOnlineStreamReady(handle.recognizer, handle.stream)) {
        DecodeOnlineStream(handle.recognizer, handle.stream);
    }

    const SherpaOnnxOnlineRecognizerResult *r =
        GetOnlineStreamResult(handle.recognizer, handle.stream);
    DestroyOnlineRecognizerResult(r);

    DestroyOnlineStream(handle.stream);
    DestroyOnlineRecognizer(handle.recognizer);
}