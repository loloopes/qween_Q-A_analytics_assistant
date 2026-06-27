.PHONY: up down logs ps

up:
	bash scripts/compose.sh up -d --build

down:
	bash scripts/compose.sh down

logs:
	bash scripts/compose.sh logs -f llm-api llm-langgraph-api

ps:
	bash scripts/compose.sh ps
