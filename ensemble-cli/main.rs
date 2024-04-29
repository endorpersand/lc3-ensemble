use std::borrow::Cow;
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use ariadne::{ColorGenerator, Label, Report, ReportKind, Source};
use clap::{Parser, Subcommand};
use lc3_ensemble::asm::{assemble, assemble_debug, ObjectFile};
use lc3_ensemble::err::ErrSpan;
use lc3_ensemble::parse::parse_ast;
use lc3_ensemble::sim::{SimErr, SimFlags, Simulator};

#[derive(Parser)]
#[command(version, about, long_about = None)]
struct Args {
    #[command(subcommand)]
    cmd: Command
}
#[derive(Subcommand)]
enum Command {
    /// Assemble an assembly source file into an object file.
    Assemble {
        /// The input assembly source file.
        input: PathBuf,
        /// The output object file.
        #[arg(short, long)]
        output: Option<PathBuf>,
        /// Whether to save debug symbols to the object file.
        #[arg(short = 'N', long)]
        no_debug_symbols: bool
    },
    /// Runs an object file.
    Run {
        input: PathBuf,

        #[arg(short, long)]
        strict: bool
    },
}

struct SourceMetadata<'fp> {
    name: &'fp str,
    src: Source<String>
}

fn main() -> ExitCode {
    let Args { cmd } = Args::parse();

    let result = match cmd {
        Command::Assemble { input, output, no_debug_symbols } => {
            cmd_assemble(
                &input, 
                output.as_deref().unwrap_or(&input.with_extension("obj")), 
                no_debug_symbols
            )
        },
        Command::Run { input, strict } => cmd_run(&input, strict)
    };

    match result {
        Ok(_)  => ExitCode::SUCCESS,
        Err(e) => e,
    }
}

fn cmd_assemble(input: &Path, output: &Path, no_debug: bool) -> Result<(), ExitCode> {
    let src = handle_read(input, std::fs::read_to_string)?;

    let meta = SourceMetadata {
        name: file_name(input).unwrap_or(""),
        src: Source::from(src.clone())
    };

    let ast = parse_ast(&src)
        .map_err(|e| report_error(e, &meta))?;
    let obj = match no_debug {
        false => assemble_debug(ast, &src),
        true  => assemble(ast),
    }.map_err(|e| report_error(e, &meta))?;

    std::fs::write(output, obj.write_bytes())
        .map_err(|e| report_simple(output, e))
}

fn cmd_run(obj_input: &Path, strict: bool) -> Result<(), ExitCode> {
    let bytes = handle_read(obj_input, std::fs::read)?;
    let obj = ObjectFile::read_bytes(&bytes)
        .ok_or_else(|| report_simple(obj_input, "could not parse object file"))?;

    let mut sim = Simulator::new(Default::default());
    sim.load_os();
    sim.load_obj_file(&obj);
    sim.flags = SimFlags {
        strict,
        ..Default::default()
    };

    sim.run()
        .map_err(|e| ReportSimErr::new(&sim, &obj, e))
        .map_err(|e| report_error(e, &SourceMetadata {
            name: file_name(obj_input).unwrap_or(""),
            src: Source::from({
                obj.symbol_table()
                    .and_then(|sym| sym.source_info())
                    .map_or_else(String::new, |s| s.source().to_string())
                })
        }))?;
    
    Err(ExitCode::FAILURE)
}

fn handle_read<'p, T>(input: &'p Path, read: impl FnOnce(&'p Path) -> std::io::Result<T>) -> Result<T, ExitCode> {
    read(input).map_err(|e| report_simple(input, e))
}

fn file_name(fp: &Path) -> Option<&str> {
    fp.file_name()?.to_str()
}
fn report_simple(fp: &Path, err: impl std::fmt::Display) -> ExitCode {
    ReportContents {
        err: &err,
        filename: file_name(fp),
        source: None,
        span: None,
        help: None,
        msg_includes_fname: true
    }.report()
}
fn report_error<E: lc3_ensemble::err::Error>(err: E, meta: &SourceMetadata) -> ExitCode {
    let span = err.span();
    let help = err.help();

    ReportContents {
        err: &err,
        filename: Some(meta.name),
        source: Some(meta.src.clone()),
        span,
        help: help.as_deref(),
        msg_includes_fname: false,
    }.report()
}

#[derive(Debug)]
struct ReportSimErr {
    kind: SimErr,
    pc: u16,
    span: Option<Range<usize>>
}
impl std::fmt::Display for ReportSimErr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} [PC = x{:04X}]", self.kind, self.pc)
    }
}
impl std::error::Error for ReportSimErr {}
impl lc3_ensemble::err::Error for ReportSimErr {
    fn span(&self) -> Option<lc3_ensemble::err::ErrSpan> {
        self.span
            .clone()
            .map(Into::into)
    }

    fn help(&self) -> Option<Cow<str>> {
        match self.kind {
            SimErr::IllegalOpcode       => None,
            SimErr::InvalidInstrFormat  => None,
            SimErr::PrivilegeViolation  => Some("RTI can only be called in supervisor mode".into()),
            SimErr::AccessViolation     => None,
            SimErr::ProgramHalted       => Some("you should not see this error. uh oh.".into()),
            SimErr::StrictRegSetUninit  => None,
            SimErr::StrictMemSetUninit  => None,
            SimErr::StrictIOSetUninit   => None,
            SimErr::StrictJmpAddrUninit => None,
            SimErr::StrictMemAddrUninit => None,
            SimErr::StrictPCCurrUninit  => Some("ensure there is a HALT instruction after your code".into()),
            SimErr::StrictPCNextUninit  => None,
            SimErr::StrictPSRSetUninit  => None,
        }
    }
}
impl ReportSimErr {
    fn new(sim: &Simulator, obj: &ObjectFile, kind: SimErr) -> Self {
        let pc = sim.prefetch_pc();
        let mut span = None;

        'sym_ref: {
            let Some(sym) = obj.symbol_table() else { break 'sym_ref };
            let Some(pc_source) = sym.find_line_source(pc) else { break 'sym_ref };
            let Some(src_info) = sym.source_info() else { break 'sym_ref };
            span = src_info.line_span(pc_source);
        }
        Self {
            kind, pc, span
        }
    }
}

struct ReportContents<'c, E> {
    err: &'c E,
    filename: Option<&'c str>,
    source: Option<Source<String>>,
    span: Option<ErrSpan>,
    help: Option<&'c str>,
    msg_includes_fname: bool
}
impl<E: std::fmt::Display> ReportContents<'_, E> {
    fn report(self) -> ExitCode {
        let mut colors = ColorGenerator::new();

        let msg = if self.msg_includes_fname {
            if let Some(fname) = self.filename {
                format!("{}: {}", fname, self.err)
            } else {
                self.err.to_string()
            }
        } else {
            self.err.to_string()
        };
        let fname = self.filename.unwrap_or("source");
        let offset = self.span.as_ref().map_or(0, |e| e.first().start);
        
        let mut report = Report::build(ReportKind::Error, fname, offset).with_message(msg);
        match self.span {
            Some(ErrSpan::One(r)) => {
                report.add_label({
                    let mut label = Label::new((fname, r))
                        .with_color(colors.next());
                    
                    if let Some(help) = self.help {
                        label = label.with_message(help);
                    }

                    label
                })
            },
            Some(ErrSpan::Two([r0, r1])) => {
                report.add_label({
                    Label::new((fname, r0))
                            .with_color(colors.next())
                            .with_message("")
                });
                report.add_label({
                    Label::new((fname, r1))
                            .with_color(colors.next())
                            .with_message("")
                });

                if let Some(help) = self.help {
                    report.set_help(help);
                }
            },
            Some(ErrSpan::Many(mr)) => {
                report.add_labels({
                    mr.into_iter()
                        .map(|s| {
                            Label::new((fname, s.clone()))
                                .with_color(colors.next())
                                .with_message("")
                        })
                });

                if let Some(help) = self.help {
                    report.set_help(help);
                }
            },
            None => {
                if let Some(help) = self.help {
                    report.add_label(Label::new((fname, 0..0)));
                    report.set_help(help);
                };
            }
        }
        
        report.finish()
            .eprint((fname, self.source.unwrap_or_else(|| Source::from(String::new()))))
            .unwrap();
        
        ExitCode::FAILURE
    }
}