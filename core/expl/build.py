from langchain_openai import ChatOpenAI


def build_generator(config):
    impl, runtime = config.impl, config.runtime

    if impl.kind == 'commercial':
        if impl.provider == 'openai':
            if "gpt-5" in impl.name:
                return ChatOpenAI(
                    model=impl.name,
                    api_key=impl.api_key,
                    reasoning={"effort": runtime.reasoning_effort},
                )
            else:
                return ChatOpenAI(
                    model=impl.name, 
                    api_key=impl.api_key, 
                    temperature=runtime.temperature
                )

    raise ValueError(f'{impl.provider}:{impl.name}은 지원되지 않습니다.')