//! Local tool (not upstream): re-translate dumped console containers
//! (`VICTORY_DUMP_ALL`) and write each host DXIL container next to them.
//! usage: retranslate <in dir> <out dir> [name filter]
//! Prints `translated=N failed=M` and, per failure reason, a count.
fn main() {
    let args: Vec<String> = std::env::args().collect();
    let (input, out) = (&args[1], &args[2]);
    let filter = args.get(3).cloned().unwrap_or_default();
    std::fs::create_dir_all(out).unwrap();
    let (mut ok, mut failed) = (0, 0);
    let mut reasons: std::collections::BTreeMap<String, u32> = Default::default();
    let mut list = String::new();
    for entry in std::fs::read_dir(input).unwrap() {
        let path = entry.unwrap().path();
        let name = path.file_name().unwrap().to_string_lossy().to_string();
        if !name.ends_with(".dxbc") || !name.contains(&filter) {
            continue;
        }
        let bytes = std::fs::read(&path).unwrap();
        let result = std::panic::catch_unwind(|| victory_shader::emit::translate(&bytes));
        match result {
            Ok(Ok(host)) => {
                std::fs::write(std::path::Path::new(out).join(&name), &host.dxil).unwrap();
                ok += 1;
            }
            Ok(Err(e)) => {
                failed += 1;
                let text = e.to_string();
                // Drop the code offset so reasons group.
                let reason = match text.find(": ") {
                    Some(i) if text[..i].contains("code at") => text[i + 2..].to_string(),
                    _ => text.clone(),
                };
                let stage = name.split('-').next().unwrap_or("?").to_string();
                *reasons.entry(format!("{stage}: {reason}")).or_default() += 1;
                list.push_str(&format!("{name}\t{reason}\n"));
            }
            Err(_) => {
                failed += 1;
                *reasons.entry("PANIC".into()).or_default() += 1;
                list.push_str(&format!("{name}\tPANIC\n"));
            }
        }
    }
    std::fs::write(std::path::Path::new(out).join("failures.tsv"), list).unwrap();
    println!("translated={ok} failed={failed}");
    let mut v: Vec<_> = reasons.into_iter().collect();
    v.sort_by(|a, b| b.1.cmp(&a.1));
    for (r, n) in v {
        println!("  {n:4} {r}");
    }
}
