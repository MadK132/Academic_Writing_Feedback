import pandas as pd

# === 1. Загружаем оба исходных датасета ===
# AI_Human.csv содержит столбцы: ['text', 'generated'] (1 = AI, 0 = Human)
# IELTS_train.csv содержит столбцы: ['prompt', 'essay', 'evaluation', 'band']
ai_human_path = "AI_Human.csv"
ielts_path = "IELTS_Cleaned.csv"

df_ai = pd.read_csv(ai_human_path)
df_ielts = pd.read_csv(ielts_path)

print(f"✅ AI/Human dataset loaded: {df_ai.shape[0]} rows")
print(f"✅ IELTS dataset loaded: {df_ielts.shape[0]} rows")

# === 2. Берём 10 000 AI-текстов ===
ai_texts = df_ai[df_ai['generated'] == 1].sample(10000, random_state=42)
ai_texts = ai_texts[['text', 'generated']]

# === 3. Берём 5 000 Human-текстов из того же набора ===
human_texts = df_ai[df_ai['generated'] == 0].sample(5000, random_state=42)
human_texts = human_texts[['text', 'generated']]

# === 4. Берём 5 000 IELTS эссе (человеческие тексты) ===
ielts_texts = df_ielts.sample(5000, random_state=42)
ielts_texts = ielts_texts.rename(columns={'essay': 'text'})
ielts_texts['generated'] = 0  # human-written
ielts_texts = ielts_texts[['text', 'generated']]

# === 5. Объединяем всё вместе ===
combined = pd.concat([ai_texts, human_texts, ielts_texts],
                     ignore_index=True)

# === 6. Перемешиваем финальный датасет ===
combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)

# === 7. Проверяем распределение классов ===
print("\n📊 Class distribution:")
print(combined['generated'].value_counts())

# === 8. Сохраняем финальный датасет ===
output_path = "combined_dataset.csv"
combined.to_csv(output_path, index=False, encoding="utf-8")

print(f"\n✅ Combined dataset saved successfully as '{output_path}'")
print(f"Total rows: {combined.shape[0]}")
print(combined.head())
