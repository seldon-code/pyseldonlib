import csv

input_file = '/home/parrot_user/Desktop/pyseldonlib/examples/ouput_20_agents/network_0.txt'  # Your CSV file
output_file ='/home/parrot_user/Desktop/pyseldonlib/examples/ouput_20_agents/network.csv'   # Output file for Cytoscape

with open(input_file, 'r') as csvfile, open(output_file, 'w', newline='') as out_csv:
    reader = csv.reader(csvfile)
    writer = csv.writer(out_csv)
    writer.writerow(['Source', 'Target', 'Weight'])  # Column names for Cytoscape
    
    # Skip the header row
    next(reader)
    
    for row in reader:
        idx_agent = row[0].strip()
        n_neighbours = int(row[1].strip())
        indices_neighbours = row[2:2 + n_neighbours]
        weights = row[2 + n_neighbours:2 + 2 * n_neighbours]

        for target, weight in zip(indices_neighbours, weights):
            writer.writerow([idx_agent, target.strip(), weight.strip()])

print("Edge list saved to edges.csv")