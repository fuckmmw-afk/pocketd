import Foundation
import Testing
@testable import PocketdKit

@Suite("Hugging Face search: links and repositories named exactly")
struct HubLookupTests {
    private let budget = DeviceBudget(
        physicalMemoryBytes: 8 * 1024 * 1024 * 1024, hasIncreasedMemoryLimit: true, environment: .device
    )

    private func details(
        _ id: String, pipeline: String?, tags: [String], gguf: Bool
    ) -> HuggingFaceSearch.Details {
        HuggingFaceSearch.Details(
            repository: .init(id: id, downloads: 1, likes: 1, gated: false, pipelineTag: pipeline, tags: tags),
            hasGGUF: gguf,
            formats: HuggingFaceSearch.weightFormats.filter { tags.contains($0) }
        )
    }

    @Test("a pasted link becomes the repository id it names")
    func links() {
        let id = "unsloth/Qwen3-4B-GGUF"
        for link in [
            "https://huggingface.co/unsloth/Qwen3-4B-GGUF",
            "  huggingface.co/unsloth/Qwen3-4B-GGUF/  ",
            "https://huggingface.co/unsloth/Qwen3-4B-GGUF/tree/main",
            "https://huggingface.co/unsloth/Qwen3-4B-GGUF/blob/main/Qwen3-4B-Q4_K_M.gguf",
            "https://hf.co/unsloth/Qwen3-4B-GGUF?library=llama-cpp",
            "https://huggingface.co/models/unsloth/Qwen3-4B-GGUF",
            "hf://models/unsloth/Qwen3-4B-GGUF",
        ] {
            #expect(HuggingFaceSearch.normalizedQuery(link) == id, "\(link)")
        }
        #expect(HuggingFaceSearch.normalizedQuery(" qwen3 4b ") == "qwen3 4b")
    }

    @Test("only owner/name is a repository id")
    func repositoryIDs() {
        #expect(HuggingFaceSearch.repositoryID(in: "FluidInference/parakeet-tdt-0.6b-v3-coreml") != nil)
        #expect(HuggingFaceSearch.repositoryID(in: "qwen3") == nil)
        #expect(HuggingFaceSearch.repositoryID(in: "qwen3 4b/x") == nil)
        #expect(HuggingFaceSearch.repositoryID(in: "a/b/c") == nil)
        #expect(HuggingFaceSearch.repositoryID(in: "/b") == nil)
    }

    @Test("a Core ML speech model says what it is, not \"nothing\"")
    func coreMLSpeech() {
        let reason = HuggingFaceSearch.whyNotOffered(details(
            "FluidInference/parakeet-tdt-0.6b-v3-coreml", pipeline: "automatic-speech-recognition", tags: ["coreml"], gguf: false
        ), for: budget)
        #expect(reason == "This is a speech recognition model in Core ML format. Pocketd runs GGUF chat models with llama.cpp, so it can't use this one.")
    }

    @Test("a chat model without GGUF files points at a conversion")
    func safetensorsChat() {
        let reason = HuggingFaceSearch.whyNotOffered(details(
            "Qwen/Qwen3-4B", pipeline: "text-generation", tags: ["transformers", "safetensors"], gguf: false
        ), for: budget)
        #expect(reason?.hasPrefix("This repository has no GGUF files — it's in safetensors format.") == true)
    }

    @Test("a GGUF model that is not for chat is named for what it is")
    func ggufSpeech() {
        let reason = HuggingFaceSearch.whyNotOffered(details(
            "ggml-org/whisper-tiny-GGUF", pipeline: "automatic-speech-recognition", tags: ["gguf"], gguf: true
        ), for: budget)
        #expect(reason == "This is a speech recognition model, not a chat model Pocketd can talk to.")
    }

    @Test("a GGUF chat model that fits is offered")
    func offered() {
        #expect(HuggingFaceSearch.whyNotOffered(details(
            "unsloth/Qwen3-4B-GGUF", pipeline: "text-generation", tags: ["gguf"], gguf: true
        ), for: budget) == nil)
        #expect(HuggingFaceSearch.whyNotOffered(details(
            "unsloth/Qwen3-235B-A22B-GGUF", pipeline: "text-generation", tags: ["gguf"], gguf: true
        ), for: budget) == "Even its smallest useful file is too big for this iPhone.")
    }
}
