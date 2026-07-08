.PHONY: build release package clean test verify

build:
	cargo build

release:
	cargo build --release
	python3 scripts/package_extension.py

verify:
	python3 scripts/verify_extension.py

package: build
	python3 scripts/package_extension.py

test:
	cargo test --lib

clippy:
	cargo clippy -- -D warnings

all: clippy test release verify
	@echo "✅ All checks passed"

clean:
	cargo clean
	rm -f target/duckdb_ml.duckdb_extension
