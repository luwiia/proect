import pandas as pd
import requests
import re

# === Чтение файла с новостями ===
news_df = pd.read_csv('news.csv', sep=';', quotechar='"', on_bad_lines='skip')

# === Параметры API для трёх моделей ===

API_CONFIG = {
    # 1) Sonar (Perplexity)
    'pplx_sonar': {
        'url': 'https://api.perplexity.ai/chat/completions',
        'model': 'sonar',
        'key': 'XXX',
        'headers': {
            'Authorization': 'Bearer {key}',
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        }
    },
    # 3) DeepSeek через OpenRouter
    'deepseek_openrouter': {
        'url': 'https://openrouter.ai/api/v1/chat/completions',
        'model': 'deepseek/deepseek-chat',
        'key': 'XXX',
        'headers': {
            'Authorization': 'Bearer {key}',
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        }
    },

}
# === Промпты ===
PROMPTS = {
    'zero_shot': """Проанализируй новостной текст и извлеки ключевые события в формате:
субъект: ...; объект: ...; действие: ...; время: ...
Если элемент не указан - напиши "не указано".

Текст: {news_text}""",

    'few_shot': """Примеры:
Текст: "Президент России Владимир Путин обсудил двусторонние отношения."
субъект: Владимир Путин; объект: двусторонние отношения; действие: обсудил; время: не указано.

Текст: "Роскомнадзор ограничил доступ к Roblox."
субъект: Роскомнадзор; объект: Roblox; действие: ограничил доступ; время: не указано.

Текст: {news_text}
""",

    'chain_of_thought': """Шаг 1: Определи ключевые события в тексте.
Шаг 2: Для каждого события найди: субъект (кто), объект (что), действие (что сделал), время.
Шаг 3: Запиши в формате: субъект: ...; объект: ...; действие: ...; время: ...

Текст: {news_text}
"""
}

# === Функция отправки запроса к API ===
def send_to_api(prompt, api_config):
    headers = {k: v.format(key=api_config['key']) for k, v in api_config['headers'].items()}
    payload = {
        "model": api_config['model'],
        "stream": False,
        "max_tokens": 500,
        "temperature": 0.2,
        "messages": [
            {"role": "system", "content": "Извлекай события строго в заданном формате."},
            {"role": "user", "content": prompt}
        ]
    }
    try:
        resp = requests.post(api_config['url'], headers=headers, json=payload)
        print("RAW:", api_config['model'], resp.status_code)  # можно оставить для отладки
        resp.raise_for_status()
        data = resp.json()
        if data.get("choices"):
            return data["choices"][0]["message"]["content"]
        return "ошибка: нет ответа от модели"
    except Exception as e:
        print("Ошибка API:", e)
        return "ошибка"

# === Парсинг ответа модели ===
def parse_model_response(response_text):
    events = []
    lines = response_text.strip().split('\n')
    for line in lines:
        if any(word in line.lower() for word in ['субъект', 'объект', 'действие', 'время']):
            subject = re.search(r'(субъект|subject)[^:]*:\s*([^;]+)', line)
            obj = re.search(r'(объект|object)[^:]*:\s*([^;]+)', line)
            action = re.search(r'(действие|action)[^:]*:\s*([^;]+)', line)
            time_match = re.search(r'(время|time)[^:]*:\s*([^;\n]+)', line)

            event = {
                'subject': subject.group(2).strip() if subject else 'не указано',
                'object': obj.group(2).strip() if obj else 'не указано',
                'action': action.group(2).strip() if action else 'не указано',
                'time': time_match.group(2).strip() if time_match else 'не указано'
            }
            events.append(event)
    return events

# === Основной цикл обработки новостей ===
def process_news(api_name):
    results = []
    api_config = API_CONFIG[api_name]
    for idx, row in news_df.iterrows():
        news_id = row['news_id']
        news_text = row['news_text']
        print(f"Обрабатывается новость {news_id}...")
        for prompt_name, prompt_template in PROMPTS.items():
            prompt = prompt_template.format(news_text=news_text)
            try:
                model_response = send_to_api(prompt, api_config)
                print(f"Ответ API ({api_name}, {prompt_name}): {model_response[:200]}...")
                parsed_events = parse_model_response(model_response)
                results.append({
                    'news_id': news_id,
                    'prompt_type': prompt_name,
                    'model_response': model_response,
                    'parsed_events': parsed_events
                })
                print(f"  ✅ {prompt_name}: найдено {len(parsed_events)} событий")
            except Exception as e:
                print(f"  ❌ {prompt_name}: ошибка - {e}")
    results_df = pd.DataFrame(results)
    results_df.to_csv(f'model_results_{api_name}.csv', index=False, sep=';', quoting=1)
    print(f"Результаты сохранены в model_results_{api_name}.csv")
    return results_df

# === Запуск для трёх моделей ===
if __name__ == "__main__":
    for api_name in API_CONFIG.keys():
        print(f"\n\n--- Обработка через {api_name} ---\n")
        process_news(api_name)
