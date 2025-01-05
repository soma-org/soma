use clap::{Parser, Subcommand};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand, Debug)]
pub enum Commands {
    /// Start an experiment
    Start {
        #[arg(long)]
        python_project_root: String,
        #[arg(long)]
        entry_point: String,
        #[arg(long)]
        port: i16,
    },
}
