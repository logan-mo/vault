import openai 
import toml

config = toml.load('app/.streamlit/secrets.toml')
openai.api_key = config['OPENAI_API_KEY']

completion = openai.chat.completions.create(
  model="gpt-3.5-turbo", # gpt-4o-2024-05-13
  messages=[
    {"role": "system", "content": "You are a poetic assistant, skilled in explaining complex programming concepts with creative flair."},
    {"role": "user", "content": "Compose a poem that explains the concept of recursion in programming."}
  ]
)

print(completion.choices[0].message.content)