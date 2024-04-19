GCC = gcc
RM = rm -rf
BIN = libsherpa.so
TEST = sherpa_test

all: clean build install

build:
	$(GCC) -shared -o $(BIN) $(shell pkg-config --cflags sherpa-onnx) \
	sherpa.c $(shell pkg-config --libs sherpa-onnx)

clean:
	$(RM) $(BIN) ../stt_engine/$(BIN) $(TEST)

install:
	cp $(BIN) ../stt_engine/$(BIN)

test:
	$(GCC) -o $(TEST) $(shell pkg-config --cflags sherpa-onnx) \
	test.c sherpa.c $(shell pkg-config --libs sherpa-onnx)
