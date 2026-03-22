
# Testing
To ensure consistent and working installation.
Testing Scripts are arranged in the order along the fallback design. Such, that each following script is dependent on the success of the previous:
1. physics : operators' physical properties verification
2. simulation : executing quantum chip dynamics
3. consistency : comparing different backends and encoding provided by DeS-ScQ
4. comparison : comparing simulation results from contemporary libraries
5. optimization : verification of circuits variation under optimization

# Benchmarking
Compare computation resource for different architechtures.
Benchmarking routines are distributed under various testing scripts.
Depending upon the objective of script performance comparison is returned.
