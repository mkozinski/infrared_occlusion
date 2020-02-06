set ylabel "recall"
set xlabel "precision"
set key outside
set terminal png 
set output "precision_recall.png"
set size square 
set grid
plot "log_v1/output_best_v2_pr.txt" u 2:1 t "input intensity", "log_v1/output_best_v2_pr.txt" u 4:3 t "neural network"
