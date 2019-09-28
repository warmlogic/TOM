# Basic commands for this repo

.PHONY: default
default: list;

# From https://stackoverflow.com/questions/4219255/how-do-you-get-the-list-of-targets-in-a-makefile
.PHONY: list
list:
	@$(MAKE) -pRrq -f $(lastword $(MAKEFILE_LIST)) : 2>/dev/null | awk -v RS= -F: '/^# File/,/^# Finished Make data base/ {if ($$1 !~ "^[#.]") {print $$1}}' | sort | egrep -v -e '^[^[:alnum:]]' -e '^$@$$' | xargs

# Build and run container

.PHONY: build
build:
	docker build -t tom .

.PHONY: run
run:
	docker run -v ~/datasets:/datasets -p 80:5000 tom --config_filepath=config.ini

# Requires running containers

.PHONY: attach
attach:
	docker exec -it tom bash

.PHONY: kill-all
kill-all:
	docker kill $(docker ps -q)

.PHONY: clean
clean:
	docker system prune
