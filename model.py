from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import time
import re


device = "cuda" if torch.cuda.is_available() else "cpu"
model_path = "feedback_gen_model"

tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path, local_files_only=True).to(device)

print("✅ Model loaded successfully!")



def trim_to_sentence_end(text: str) -> str:
    """Обрезает текст по последнему завершённому предложению."""
    match = re.search(r"([.!?])[^.!?]*$", text)
    return text[:match.end(1)] if match else text



def clean_feedback(text: str) -> str:
    """
    Полная очистка фидбэка:
    - убирает 'Band Score', скобки и разметку;
    - исправляет частые опечатки и дубли;
    - возвращает грамматически правильный, цельный отзыв.
    """
    import re
    original_text = text.strip()

    text = re.sub(r"Suggested\s*Band\s*Score\s*\([^)]*\)", "", text, flags=re.I)
    text = re.sub(r"\(Vocabulary\):?,?", "", text, flags=re.I)
    text = re.sub(r"Overall\s*Band\s*Score[:\s\.0-9]*", "", text, flags=re.I)
    text = re.sub(r"Feedback\s*and\s*Additional.*?:", "", text, flags=re.I)
    text = re.sub(r"###|##|\*\*|__", "", text)
    text = re.sub(r"[-–•]\s*", "", text)
    text = re.sub(r"\s{2,}", " ", text).strip()

    fixes = {
        "a some": "some",
        "grammarulation": "grammar and formulation",
        "ecommerce": "e-commerce",
        "coherence and grammarulation": "coherence and grammar",
    }
    for wrong, right in fixes.items():
        text = text.replace(wrong, right)

    sentences = re.split(r'(?<=[.!?])\s+', text)
    unique = []
    for s in sentences:
        s_clean = s.strip()
        if not s_clean:
            continue
        if any(s_clean.lower() in u.lower() or u.lower() in s_clean.lower() for u in unique):
            continue
        unique.append(s_clean)
    text = " ".join(unique)

    text = re.sub(r"\s+", " ", text).strip()
    if text and text[0].islower():
        text = text[0].upper() + text[1:]
    if not text.endswith(('.', '!', '?')):
        text += '.'

    return text or original_text

def remove_scores(text: str) -> str:
    """Удаляет все числовые оценки (6, 6.5, 7.0, 8.5 и т.п.) из текста."""
    import re
    text = re.sub(r"(Band\s*Score[:=]?\s*)?\b\d+(\.\d+)?\b", "", text, flags=re.I)
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"\s+([.,])", r"\1", text)
    return text.strip()


def light_clean_feedback(text: str) -> str:
    """Удаляет оценки, символы форматирования и вспомогательные маркеры."""
    import re
    text = re.sub(r"\b\d+(\.\d+)?\b", "", text)  
    text = re.sub(r"Suggested\s*Band\s*Score[:=]?", "", text, flags=re.I)
    text = re.sub(r"Lexical Resource\s*\(.*?\)\s*:", "Lexical Resource:", text, flags=re.I)
    text = re.sub(r"Feedback and Additional Comments:", "", text, flags=re.I)
    text = re.sub(r"Suggestions for Improvement:", "", text, flags=re.I)
    text = re.sub(r"\*", "", text) 
    text = re.sub(r"\s{2,}", " ", text).strip()
    return text




def generate_feedback(essay_text: str) -> str:
    """Генерация и автоматическая очистка фидбэка."""
    model.eval()
    input_text = essay_text.strip()

    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=768,
    ).to(device)

    start_time = time.time()
    outputs = model.generate(
        **inputs,
        max_new_tokens=220,
        min_length=80,
        num_beams=4,
        early_stopping=True,
        no_repeat_ngram_size=3,
        do_sample=False,       
    )

    feedback = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    feedback = trim_to_sentence_end(feedback)
    feedback = clean_feedback(feedback)
    feedback = remove_scores(feedback)

    print(f"⏱ Generated in {time.time() - start_time:.2f}s")
    return feedback



print("\n Feedback Generator Ready! Type 'exit' to stop.\n")

while True:
    essay_text = input(" Enter essay text: ").strip()
    if essay_text.lower() == "exit":
        break

    print("\n⏳ Generating feedback...")
    feedback = generate_feedback(essay_text)
    print("\n Feedback:\n", feedback)
    print("-" * 80)

