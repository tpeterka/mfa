.ONESHELL:
ROOT_DIR:=$(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))


pybind: ## compile and build the python bindings
	rm -rf build
	mkdir build
	cd build; cmake $(ROOT_DIR) -Dpython=true -DCMAKE_BUILD_TYPE=Debug
	$(MAKE) -C $(ROOT_DIR)/build/python
	@echo $(ROOT_DIR)


.PHONY: help

help: ## help command
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'


.DEFAULT_GOAL := help
