import math
import ollama

EMBEDDING_MODEL = "hf.co/CompendiumLabs/bge-base-en-v1.5-gguf"
LANGUAGE_MODEL = "hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF"

VECTOR_DB = []


def load_dataset(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return [line.strip() for line in file if line.strip()]


def add_chunk_to_database(chunk):
    response = ollama.embed(model=EMBEDDING_MODEL, input=chunk)
    VECTOR_DB.append((chunk, response["embeddings"][0]))


def cosine_similarity(a, b):
    if not a or not b or len(a) != len(b):
        return 0.0
    dot_product = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot_product / (norm_a * norm_b)


def retrieve(query, top_n=3):
    query_embedding = ollama.embed(model=EMBEDDING_MODEL, input=query)["embeddings"][0]

    similarities = []
    for chunk, embedding in VECTOR_DB:
        score = cosine_similarity(query_embedding, embedding)
        similarities.append((chunk, score))

    similarities.sort(key=lambda item: item[1], reverse=True)
    return similarities[:top_n]


def main():
    dataset = load_dataset("cat-facts.txt")

    print("Step 1 Checkpoint")
    print(f"Total chunks: {len(dataset)}")
    if len(dataset) >= 2:
        print(f"Chunk 1: {dataset[0]}")
        print(f"Chunk 2: {dataset[1]}\n")

    for chunk in dataset:
        add_chunk_to_database(chunk)

    print("Step 2 Checkpoint")
    print(f"len(VECTOR_DB) == len(dataset): {len(VECTOR_DB) == len(dataset)}")
    is_list_of_numbers = isinstance(VECTOR_DB[0][1], list) and isinstance(VECTOR_DB[0][1][0], float)
    print(f"Embedding is a list of numbers: {is_list_of_numbers}\n")

    print("Step 3 Checkpoint")
    vec_a, vec_b, vec_c = [1, 0, 0], [1, 0, 0], [0, 1, 0]
    print(f"Test 1 (Identical vectors): {cosine_similarity(vec_a, vec_b)}")
    print(f"Test 2 (Orthogonal vectors): {cosine_similarity(vec_a, vec_c)}\n")

    print("Step 4 Checkpoint")
    test_queries = ["How long do cats sleep?", "What is a bezoar?"]
    for q in test_queries:
        print(f"Query: {q}")
        results = retrieve(q, top_n=2)
        for chunk, score in results:
            print(f"  ({score:.3f}) {chunk}")
    print("\nStep 6 Checkpoint (Pipeline Interaction)")

    while True:
        input_query = input("\nAsk a question (or type 'exit' to quit): ")
        if input_query.lower() == 'exit':
            break

        retrieved_knowledge = retrieve(input_query, top_n=3)

        print("\nRetrieved knowledge:")
        for chunk, similarity in retrieved_knowledge:
            print(f"- ({similarity:.3f}) {chunk}")

        context = "\n".join(f"- {chunk}" for chunk, _similarity in retrieved_knowledge)
        instruction_prompt = f"""You are a grounded question-answering assistant.
Use only the context below to answer the user's question.
If the context does not contain enough evidence, say that the answer is not in the knowledge base.
When records conflict, prefer a clearly dated newer record and explain the update briefly.

Context:
{context}
"""
        print("\nAnswer:")
        stream = ollama.chat(
            model=LANGUAGE_MODEL,
            messages=[
                {"role": "system", "content": instruction_prompt},
                {"role": "user", "content": input_query},
            ],
            stream=True,
        )
        for response_chunk in stream:
            print(response_chunk["message"]["content"], end="", flush=True)
        print()


if __name__ == "__main__":
    main()
    