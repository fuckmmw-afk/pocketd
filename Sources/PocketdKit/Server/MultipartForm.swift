import Foundation

/// `multipart/form-data`, which is how OpenAI's transcription endpoint takes
/// its audio — and therefore how every client of it sends one.
struct MultipartForm: Sendable, Equatable {
    struct Part: Sendable, Equatable {
        var name: String
        var filename: String?
        var contentType: String?
        var data: Data

        var text: String { String(decoding: data, as: UTF8.self) }
    }

    var parts: [Part]

    func part(named name: String) -> Part? { parts.first { $0.name == name } }
    func value(_ name: String) -> String? {
        part(named: name).map { $0.text.trimmingCharacters(in: .whitespacesAndNewlines) }
    }

    enum ParseError: Error, Equatable {
        case notMultipart
        case malformed
    }

    /// The boundary from a `Content-Type` header, quoted or not.
    static func boundary(from contentType: String?) -> String? {
        guard let contentType, contentType.lowercased().hasPrefix("multipart/form-data") else { return nil }
        for parameter in contentType.split(separator: ";").dropFirst() {
            let pair = parameter.split(separator: "=", maxSplits: 1).map { $0.trimmingCharacters(in: .whitespaces) }
            guard pair.count == 2, pair[0].lowercased() == "boundary" else { continue }
            var value = pair[1]
            if value.hasPrefix("\""), value.hasSuffix("\""), value.count >= 2 {
                value = String(value.dropFirst().dropLast())
            }
            return value.isEmpty ? nil : value
        }
        return nil
    }

    /// Parts are split on bytes, never decoded as text: the audio is binary
    /// and may contain anything, including bytes that look like a boundary
    /// line's start but are not one.
    static func parse(_ body: Data, contentType: String?) throws -> MultipartForm {
        guard let boundary = boundary(from: contentType) else { throw ParseError.notMultipart }
        let delimiter = Data("--\(boundary)".utf8)
        let separator = Data("\r\n\r\n".utf8)

        var parts: [Part] = []
        guard var cursor = body.range(of: delimiter)?.upperBound else { throw ParseError.malformed }
        while cursor < body.endIndex {
            // `--` straight after a delimiter ends the form.
            if body[cursor...].starts(with: Data("--".utf8)) { break }
            guard let next = body.range(of: Data("\r\n".utf8) + delimiter, in: cursor..<body.endIndex) else {
                throw ParseError.malformed
            }
            var chunk = body[cursor..<next.lowerBound]
            if chunk.starts(with: Data("\r\n".utf8)) { chunk = chunk.dropFirst(2) }
            guard let split = chunk.range(of: separator) else { throw ParseError.malformed }
            let headers = String(decoding: chunk[chunk.startIndex..<split.lowerBound], as: UTF8.self)
            let data = Data(chunk[split.upperBound...])
            if let part = Self.part(headers: headers, data: data) { parts.append(part) }
            cursor = next.upperBound
        }
        return MultipartForm(parts: parts)
    }

    private static func part(headers: String, data: Data) -> Part? {
        var name: String?
        var filename: String?
        var contentType: String?
        for line in headers.components(separatedBy: "\r\n") {
            let pair = line.split(separator: ":", maxSplits: 1).map { $0.trimmingCharacters(in: .whitespaces) }
            guard pair.count == 2 else { continue }
            switch pair[0].lowercased() {
            case "content-disposition":
                name = parameter("name", in: pair[1])
                filename = parameter("filename", in: pair[1])
            case "content-type":
                contentType = pair[1]
            default:
                break
            }
        }
        guard let name else { return nil }
        return Part(name: name, filename: filename, contentType: contentType, data: data)
    }

    /// `name="file"` out of `form-data; name="file"; filename="a.m4a"`.
    private static func parameter(_ key: String, in header: String) -> String? {
        for item in header.split(separator: ";").map({ $0.trimmingCharacters(in: .whitespaces) }) {
            let pair = item.split(separator: "=", maxSplits: 1).map(String.init)
            guard pair.count == 2, pair[0].lowercased() == key else { continue }
            var value = pair[1]
            if value.hasPrefix("\""), value.hasSuffix("\""), value.count >= 2 {
                value = String(value.dropFirst().dropLast())
            }
            return value
        }
        return nil
    }
}
