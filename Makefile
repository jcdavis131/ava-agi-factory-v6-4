.PHONY: help images volumes up down logs ps bench train serve report test smoke clean-consumed

AVA_PRESET ?= nano
COMPOSE    := docker compose

help:
	@echo "Ava continuous pipeline"
	@echo "  make volumes        create the five external named volumes"
	@echo "  make images         build ava/cpu and ava/gpu"
	@echo "  make up             start the whole pipeline (gather+clean+train+serve)"
	@echo "  make up-data        data plane only (collector + curator)"
	@echo "  make down           stop everything"
	@echo "  make logs           follow trainer logs"
	@echo "  make ps             pipeline status + shard counts by state"
	@echo "  make bench          measure collector/curator/trainer throughput"
	@echo "  make train          one-off trainer run in the foreground"
	@echo "  make serve          serve latest checkpoint on :8000"
	@echo "  make report         regenerate reports/index.html"
	@echo "  make test           run the full test suite (in the cpu image)"
	@echo "  make smoke          end-to-end nano smoke test"

volumes:
	@for v in ava_raw ava_packed ava_ckpt ava_state ava_reports; do \
		docker volume create $$v >/dev/null && echo "ok $$v"; done

images:
	$(COMPOSE) build

up: volumes
	AVA_PRESET=$(AVA_PRESET) $(COMPOSE) up -d

up-data: volumes
	AVA_PRESET=$(AVA_PRESET) $(COMPOSE) up -d collector curator janitor

down:
	$(COMPOSE) down

logs:
	$(COMPOSE) logs -f trainer

ps:
	@$(COMPOSE) ps
	@echo "--- shards by state ---"
	@docker run --rm -v ava_state:/state ava/cpu:latest \
		python -m ava.pipeline.manifest --summary

bench:
	docker run --rm --gpus all -v ava_packed:/packed -v ava_state:/state \
		ava/gpu:latest python scripts/bench_pipeline.py --preset $(AVA_PRESET)

train:
	AVA_PRESET=$(AVA_PRESET) $(COMPOSE) run --rm trainer

serve:
	$(COMPOSE) up -d server && \
	echo "serving on http://localhost:8000  (viewer: /jspace/viewer, report: /report)"

report:
	docker run --rm -v ava_reports:/reports -v ava_ckpt:/ckpt ava/cpu:latest \
		python scripts/make_report.py --out /reports/index.html

test:
	docker run --rm -v "$(CURDIR)/tests:/app/tests:ro" ava/cpu:latest \
		python -m pytest tests/ -x -q

smoke:
	bash scripts/smoke_e2e.sh
