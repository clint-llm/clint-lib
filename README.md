# Clint Lib

This is the web UI for the [Clint LLM project](https://github.com/clint-llm/clint-llm.github.io).
Check out the [live project](https://clint-llm.github.io).

The library is written in Rust and is compiled to WASM to serve the [Clint UI](https://github.com/clint-llm/clint/clint-ui).
Rust was chosen because it can compile to a browser environment,
and because it has a good ecosystem of libraries that support the networking and numerical computations that are carried out in this library.

## Development

### Environment

- You will need Rust and cargo for package management and compilation:
  - <https://rustup.rs/>
- You will need wasm-pack to compile to a browser environment:
  - <https://rustwasm.github.io/wasm-pack/installer/>

### Package structure

- The public interface in `lib.rs` is intended for use in a JavaScript environment.
- To avoid clashes with internal names in the library,
  most names in `lib.rs` are suffixed with `Js` or `_js`.
- The `docdb` module implements a DB that can serve markdown content and search it using vector similarity.
  - This DB is loaded in the browser memory.
  - Its indices are downloaded over HTTP.
  - The documents are not stored in browser memory. They are served separately.
  - The expected number of documents in 10k to 100k.
- The `openai` module provides an interface for some of OpenAI's chat completion and embedding endpoints.
  - This isn't a complete interface to the OpenAI API.
  - This is necessary to provide streaming responses that compile to WASM.
- The `prompt` module provides functions to "call" GPT with the prompts that make up the Clint process.
  - `prompt::rewrite` rewrites a message using medical terminology.
  - `prompt::notes` uses the re-written message to write or update clinical notes.
  - `prompt::diagnosis::initial` uses the notes and retrieved documents to list plausible diagnoses
  - `prompt::diagnosis::refine` refines a single diagnosis using retrieved documents and a more precise prompt
  - `prompt::respond` responds to the last message with the notes and diagnoses as context
  - `prompt::cite` provides URLs for relevant retrieved documents

### GPT

A single user message uses several thousand tokens and makes several requests to GPT.
GPT 3.5 is used for its speed and cost when compared to GPT 4.
The entire Clint process is a quasi Retrieval Augmented Generation,
so there is less risk of hallucination and invalid reasoning form using GPT 3.5.

### Initial build setup

The project was initialized with the following commands (included for documentation):

- Setup the library using wasm-pack:
  - <https://rustwasm.github.io/docs/wasm-pack/tutorials/npm-browser-packages/index.html>
  - `wasm-pack new clint-lib`

### Commands

To build the app: `wasm-pack build`.

To format the source code: `cargo fmt`.
