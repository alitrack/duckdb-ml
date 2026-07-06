.PHONY: build release package clean test

build:
	cargo build

release:
	cargo build --release
	python3 scripts/package_extension.py

package: build
	python3 scripts/package_extension.py

test:
	cargo test

clean:
	cargo clean
	rm -f target/duckdb_ml.duckdb_extension
