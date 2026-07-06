use std::io::{Read, Write};
use std::path::Path;

#[allow(clippy::collapsible_if)]
fn main() {
    // Only run in release builds (or when explicitly requested)
    let out_dir = std::env::var("OUT_DIR").unwrap_or_default();

    // Detect DuckDB version from the system duckdb binary, or use a default
    let duckdb_version = detect_duckdb_version().unwrap_or_else(|| "v1.5.2".to_string());

    // Platform string
    let platform = format!("{}_{}", std::env::consts::OS, std::env::consts::ARCH)
        .replace("linux", "linux")
        .replace("x86_64", "amd64");

    // Write version info for the post-build script
    let version_file = Path::new(&out_dir).join("duckdb_extension_info.txt");
    let mut f = std::fs::File::create(&version_file).unwrap();
    writeln!(f, "version={}", duckdb_version).unwrap();
    writeln!(f, "platform={}", platform).unwrap();

    println!("cargo:rerun-if-changed=build.rs");
    println!(
        "cargo:rustc-env=DUCKDB_EXTENSION_VERSION={}",
        duckdb_version
    );
    println!("cargo:rustc-env=DUCKDB_EXTENSION_PLATFORM={}", platform);
}

fn detect_duckdb_version() -> Option<String> {
    // Try to get version from system duckdb
    #[allow(clippy::collapsible_if)]
    if let Ok(output) = std::process::Command::new("duckdb")
        .arg("--version")
        .output()
    {
        let stdout = String::from_utf8_lossy(&output.stdout);
        if let Some(version) = stdout.split_whitespace().next() {
            if version.starts_with('v') {
                return Some(version.to_string());
            }
        }
    }

    // Fallback: try the DuckDB C API at compile time
    // libduckdb-sys exposes the version via pkg-config or compile-time defines
    if let Ok(version) = std::env::var("DEP_DUCKDB_VERSION") {
        return Some(format!("v{}", version));
    }

    None
}

/// Append the DuckDB extension metadata footer to the compiled .so.
/// Call this from a post-build step (xtask or Makefile).
pub fn append_extension_footer(so_path: &str, ext_path: &str, version: &str, platform: &str) {
    let mut data = Vec::new();
    std::fs::File::open(so_path)
        .unwrap()
        .read_to_end(&mut data)
        .unwrap();

    let footer = build_footer(version, platform);
    data.extend_from_slice(&footer);

    std::fs::write(ext_path, &data).unwrap();
    eprintln!("Wrote extension: {} ({} bytes)", ext_path, data.len());
}

fn build_footer(version: &str, platform: &str) -> Vec<u8> {
    let mut footer = Vec::with_capacity(352);

    // Version string (32 bytes, null-padded)
    let mut version_bytes = vec![0u8; 32];
    let vb = version.as_bytes();
    let len = vb.len().min(31);
    version_bytes[..len].copy_from_slice(&vb[..len]);
    footer.extend_from_slice(&version_bytes);

    // Platform string (32 bytes, null-padded)
    let mut platform_bytes = vec![0u8; 32];
    let pb = platform.as_bytes();
    let len = pb.len().min(31);
    platform_bytes[..len].copy_from_slice(&pb[..len]);
    footer.extend_from_slice(&platform_bytes);

    // Extension ABI version (u64 LE, 0 = C API)
    footer.extend_from_slice(&0u64.to_le_bytes());

    // Reserved (24 bytes)
    footer.extend_from_slice(&[0u8; 24]);

    // Signature placeholder (256 bytes, unsigned)
    footer.extend_from_slice(&[0u8; 256]);

    assert_eq!(footer.len(), 352);
    footer
}
