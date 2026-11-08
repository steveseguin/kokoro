import { describe, test, expect, vi } from "vitest";

// These tests exercise streaming control flow, not model inference or downloads.
vi.mock("@huggingface/transformers", () => ({
  StyleTextToSpeech2Model: {}, AutoTokenizer: {}, Tensor: class {}, RawAudio: class {},
}));
vi.mock("../src/phonemize.js", () => ({ phonemize: vi.fn(async (text) => text) }));
vi.mock("../src/voices.js", () => ({ VOICES: { af_heart: {} }, getVoiceData: vi.fn() }));

import { KokoroTTS, TextSplitterStream } from "../src/kokoro.js";

function createTTS() {
  const tts = new KokoroTTS(null, (text) => ({ input_ids: text }));
  tts.generate_from_ids = vi.fn(async () => ({ audio: true }));
  return tts;
}

async function collect(stream) {
  const result = [];
  for await (const item of stream) result.push(item.text);
  return result;
}

describe("streaming finite strings", () => {
  test.each([
    ["Hello.", ["Hello."]],
    ["Hello", ["Hello"]],
    ["First sentence. Last sentence.", ["First sentence.", "Last sentence."]],
    ["First sentence. Unfinished tail", ["First sentence.", "Unfinished tail"]],
    ["First line\nLast line", ["First line", "Last line"]],
    ["", []],
    [" \n\t ", []],
  ])("emits all text and completes for %j", async (text, expected) => {
    expect(await collect(createTTS().stream(text))).toEqual(expected);
  }, 500);

  test("completes when a custom split pattern leaves no chunks", async () => {
    expect(await collect(createTTS().stream("\n\n", { split_pattern: /\n+/ }))).toEqual([]);
  }, 500);
});

describe("streaming caller-owned splitters", () => {
  test("waits for more input and lets the caller close the stream", async () => {
    const splitter = new TextSplitterStream();
    const stream = createTTS().stream(splitter);
    splitter.push("First sentence. Pending");
    expect((await stream.next()).value.text).toBe("First sentence.");

    let settled = false;
    const pending = stream.next().then((result) => { settled = true; return result; });
    await new Promise((resolve) => setTimeout(resolve, 10));
    expect(settled).toBe(false);

    splitter.push(" tail.");
    splitter.close();
    expect((await pending).value.text).toBe("Pending tail.");
    expect((await stream.next()).done).toBe(true);
  });

  test("accepts an already closed splitter", async () => {
    const splitter = new TextSplitterStream();
    splitter.push("Final sentence.");
    splitter.close();
    expect(await collect(createTTS().stream(splitter))).toEqual(["Final sentence."]);
  });
});
