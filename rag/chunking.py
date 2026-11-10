
import re


# ==========================================
# 1. SPLIT LONG SENTENCES
# ==========================================

def split_long_sentence(sentence, chunk_size):

    words = sentence.split()
    segments = []
    current = ""

    for word in words:

        if len(word) > chunk_size:
            if current:
                segments.append(current)
                current = ""

            for i in range(0, len(word), chunk_size):
                segments.append(word[i:i + chunk_size])

            continue

        candidate = (current + " " + word).strip()

        if len(candidate) <= chunk_size:
            current = candidate
        else:
            if current:
                segments.append(current)

            current = word

    if current:
        segments.append(current)

    return segments


# ==========================================
# 2. SPLIT TEXT INTO SENTENCES
# ==========================================

def split_into_sentences(text, chunk_size):

    paragraphs = re.split(r"\n\s*\n", text.strip())

    sentences = []

    for paragraph in paragraphs:

        paragraph = re.sub(r"\s+", " ", paragraph).strip()

        if not paragraph:
            continue

        parts = re.split(
            r'(?<=[.!?])\s+',
            paragraph
        )

        for part in parts:

            part = part.strip()

            if not part:
                continue

            if len(part) > chunk_size:
                sentences.extend(
                    split_long_sentence(part, chunk_size)
                )
            else:
                sentences.append(part)

    return sentences


# ==========================================
# 3. CREATE CONTEXT-AWARE CHUNKS
# ==========================================

def chunk_documents(
    documents,
    chunk_size=1000,
    overlap=150
):

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError(
            "overlap must be non-negative and smaller than chunk_size."
        )

    chunks = []

    for document in documents:

        text = document.get("text", "")
        metadata = document.get("metadata", {})

        if not text or not text.strip():
            continue

        sentences = split_into_sentences(
            text,
            chunk_size
        )

        current_chunk = []

        for sentence in sentences:

            candidate = " ".join(
                current_chunk + [sentence]
            ).strip()

            if len(candidate) <= chunk_size:

                current_chunk.append(sentence)

            else:

                if current_chunk:

                    chunk_text = " ".join(
                        current_chunk
                    ).strip()

                    chunks.append({
                        "text": chunk_text,
                        "metadata": metadata.copy()
                    })

                # Preserve context from the previous chunk
                overlap_sentences = []
                overlap_length = 0

                for previous_sentence in reversed(current_chunk):

                    if (
                        overlap_length
                        + len(previous_sentence)
                        + 1
                        <= overlap
                    ):

                        overlap_sentences.insert(
                            0,
                            previous_sentence
                        )

                        overlap_length += (
                            len(previous_sentence) + 1
                        )

                    else:
                        break

                current_chunk = overlap_sentences + [sentence]

        # Add remaining sentences
        if current_chunk:

            chunk_text = " ".join(
                current_chunk
            ).strip()

            chunks.append({
                "text": chunk_text,
                "metadata": metadata.copy()
            })

    print(f"Total chunks created: {len(chunks)}")

    return chunks
