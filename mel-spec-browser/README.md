# WASM Mel Spectrogram with Voice Activity Detection

In this example Mel WASM Worker and Audio Worklets process audio in the background
and share mel spectrogram frames with the main ui thread via a Shared Array Buffer.

It renders in real-time on an M2 Air.

## Run

Run it:

```sh
caddy run -c Caddyfile
```

## Acknowledgements

- https://github.com/wavey-ai/mel-spec
- https://github.com/wavey-ai/mel-spec/tree/main/examples/browser
