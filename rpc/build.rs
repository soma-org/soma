// build.rs
use prost_types::FileDescriptorSet;
use protox::prost::Message as _;
use std::collections::HashMap;
use std::path::PathBuf;

// Include your build utilities as modules
mod build_utils {
    pub mod generate_fields;
    pub mod generate_getters;
    pub mod pbjson_build;
}

fn main() {
    println!("cargo:rerun-if-changed=proto");
    println!("cargo:rerun-if-changed=build-utils");

    let root_dir = PathBuf::from(std::env!("CARGO_MANIFEST_DIR"));
    let proto_dir = root_dir.join("proto");

    // Output to src/proto/generated
    let out_dir = root_dir.join("src/proto/generated");
    std::fs::create_dir_all(&out_dir).expect("Failed to create output directory");

    let proto_ext = std::ffi::OsStr::new("proto");
    let proto_files = walkdir::WalkDir::new(&proto_dir)
        .into_iter()
        .filter_map(|entry| {
            (|| {
                let entry = entry?;
                if entry.file_type().is_dir() {
                    return Ok(None);
                }
                let path = entry.into_path();
                if path.extension() != Some(proto_ext) {
                    return Ok(None);
                }
                Ok(Some(path))
            })()
            .transpose()
        })
        .collect::<Result<Vec<_>, walkdir::Error>>()
        .unwrap();

    if proto_files.is_empty() {
        println!("cargo:warning=No proto files found!");
        return;
    }

    let mut fds = protox::Compiler::new(std::slice::from_ref(&proto_dir))
        .unwrap()
        .include_source_info(true)
        .include_imports(true)
        .open_files(&proto_files)
        .unwrap()
        .file_descriptor_set();

    // Sort files by name to have deterministic codegen output
    fds.file.sort_by(|a, b| a.name.cmp(&b.name));

    // Generate the main proto files with tonic
    if let Err(error) = tonic_build::configure()
        .build_client(true)
        .build_server(true)
        .bytes(["."])
        .boxed(".soma.rpc.Input.literal")
        .boxed(".soma.rpc.Epoch.system_state")
        .boxed("json")
        .message_attribute(".soma.rpc", "#[non_exhaustive]")
        .enum_attribute(".soma.rpc", "#[non_exhaustive]")
        .btree_map(["."])
        .out_dir(&out_dir)
        .compile_fds(fds.clone())
    {
        panic!("failed to compile protos: {error}");
    }

    // Group files by package
    let mut packages: HashMap<_, FileDescriptorSet> = HashMap::new();
    for mut file in fds.file {
        // Clear out the source code info as its not required for reflection
        file.source_code_info = None;
        packages
            .entry(file.package().to_owned())
            .or_default()
            .file
            .push(file);
    }

    // Generate custom field info
    build_utils::generate_fields::generate_field_info(&packages, &out_dir);

    // Generate getters
    build_utils::generate_getters::generate_getters(&packages, &out_dir);

    // Generate JSON serialization with pbjson
    let mut json_builder = build_utils::pbjson_build::Builder::new();
    for file in packages.values().flat_map(|set| set.file.iter()) {
        json_builder.register_file_descriptor(file.to_owned());
    }

    json_builder
        .out_dir(&out_dir)
        .ignore_unknown_fields()
        .btree_map(["."])
        .build(&[".google.rpc", ".soma"])
        .unwrap();

    // Store FDS files for runtime reflection
    for (package, fds) in packages {
        let file_name = format!("{package}.fds.bin");
        let file_descriptor_set_path = out_dir.join(&file_name);
        std::fs::write(file_descriptor_set_path, fds.encode_to_vec()).unwrap();
    }

    println!("cargo:warning=Proto generation complete!");
}
