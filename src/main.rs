use clap::{Arg, App}; // For command line arguments
use rayon::prelude::*; // For parallelisation
use rand::Rng; // Randomisation
use rand::SeedableRng; // Seed setting

use std::time::Instant; // Timing
use std::fs::File; // File manipulation
use std::io::{Read, Write}; // input output

fn main() -> std::io::Result<()> {
    // Command line arguments. 
    // -- -m (serial|parallell) will run the program using either serial or parallell implementations.
    // -- -c The program will save the results to a binary file by default. If you run with the -c (check) flag, it will compare results for consistancy.
    let matches = App::new("Matrix Multiplication Checker")
        .version("1.0")
        .arg(Arg::with_name("check")
            .short("c")
            .long("check")
            .help("Check the results against the binary file"))
        .arg(Arg::with_name("mode")
            .short("m")
            .long("mode")
            .takes_value(true)
            .default_value("parallel")
            .help("Mode of operation: parallel or serial"))
        .get_matches();

    // Matrix size
    let n = 1500;

    // Initializing matrices
    let mut a: Vec<Vec<f64>> = vec![vec![0.0; n]; n];
    let mut b: Vec<Vec<f64>> = vec![vec![0.0; n]; n];
    let mut c: Vec<Vec<f64>> = vec![vec![0.0; n]; n];

    let seed = [0u8; 32]; // Use a 32-byte array to seed
    let mut rng = rand::rngs::StdRng::from_seed(seed);

    // Filling matrices A and B with random numbers
    for i in 0..n {
        for j in 0..n {
            a[i][j] = rng.gen::<f64>();
            b[i][j] = rng.gen::<f64>();
        }
    }

    if matches.is_present("check") {
        // Read matrix D from binary file
        let mut file = File::open("log.bin")?;
        let mut d: Vec<Vec<f64>> = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in 0..n {
                let mut buffer = [0u8; 8];
                file.read_exact(&mut buffer)?;
                d[i][j] = f64::from_le_bytes(buffer);
            }
        }

        // Performing matrix multiplication between matrices A and B, storing the result in matrix C
        for i in 0..n {
            for j in 0..n {
                let mut total = 0.0;
                for k in 0..n {
                    total += a[i][k] * b[k][j];
                }
                c[i][j] = total;
            }
        }

        // Assert equality using binary file
        for i in 0..n {
            for j in 0..n {
                assert!(c[i][j] == d[i][j], "Mismatch at ({}, {})", i, j);
            }
        }

        println!("No mismatches found");
    } else {
        // Performing matrix multiplication between matrices A and B, storing the result in matrix C
        let mut total_duration = std::time::Duration::new(0, 0);
        let runs = 10;
        for _ in 0..runs {
            let start_time = Instant::now();
    
            match matches.value_of("mode").unwrap_or("parallel") {
                "serial" => {
                    for i in 0..n {
                        for j in 0..n {
                            let mut total = 0.0;
                            for k in 0..n {
                                total += a[i][k] * b[k][j];
                            }
                            c[i][j] = total;
                        }
                    }
                },
                "parallel" | _ => {
                    c.par_iter_mut().enumerate().for_each(|(i, row)| {
                        for j in 0..n {
                            let mut total = 0.0;
                            for k in 0..n {
                                total += a[i][k] * b[k][j];
                            }
                            row[j] = total;
                        }
                    });
                },
            }

            total_duration += start_time.elapsed();
            println!("Run time taken: {:?}", start_time.elapsed());
        }


        let avg_duration = total_duration / runs;
        println!("Average time taken: {:?}", avg_duration);

        // Writing matrix C to binary file
        let mut file = File::create("log.bin")?;
        // Timing 10 rounds of matrix multiplication
        for i in 0..n {
            for j in 0..n {
                file.write_all(&c[i][j].to_le_bytes())?;
            }
        }
    }
    Ok(())
}