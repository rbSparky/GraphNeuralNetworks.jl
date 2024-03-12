

hin_A, hout_A, hidden_A = 5, 5, 10
        hin_B, hout_B, hidden_B = 3, 3, 6
        num_nodes_A, num_nodes_B = 5, 3  

        hg = rand_bipartite_heterograph((num_nodes_A, num_nodes_B), 15) 

        layers = HeteroGraphConv([
            (:A, :to, :B) => EGNNConv(hin_A => hout_B),
            (:B, :to, :A) => EGNNConv(hin_B => hout_A)
        ])

        T = Float32  
        h = (A = randn(T, hin_A, num_nodes_A), B = randn(T, hin_B, num_nodes_B))
        x = (A = rand(T, 3, num_nodes_A), B = rand(T, 3, num_nodes_B)) 

        y = layers(hg, h, x)

        @test size(y[:A].h) == (hout_A, num_nodes_A)
        @test size(y[:B].h) == (hout_B, num_nodes_B)
        @test size(y[:A].x) == (3, num_nodes_A) 
        @test size(y[:B].x) == (3, num_nodes_B)


        hin = 5
        hout = 5
        hidden = 5
        hg = rand_bipartite_heterograph((2,3), 6)
        hg.num_nodes
        x = (A = rand(Float32, 4, 2), B = rand(Float32, 4, 3))
        h = (A = rand(Float32, 5, 2), B = rand(Float32, 5, 3))
        layers = HeteroGraphConv((:A, :to, :B) => EGNNConv(4 => 2),
                                 (:B, :to, :A) => EGNNConv(4 => 2));
        y = layers(hg)
        
in_channel = 3
out_channel = 5
N = 4
T = Float32

        hin = 5
        hout = 5
        hidden = 5
        l = EGNNConv(hin => hout, hidden)
        g = rand_graph(10, 20)
        x = rand(T, in_channel, g.num_nodes)
        h = randn(T, hin, g.num_nodes)
        hnew, xnew = l(g, h, x)        
        @test size(hnew) == (hout, g.num_nodes)



        x = (A = rand(4, 2), B = rand(4, 3))
        layers = HeteroGraphConv((:A, :to, :B) => GINConv(Dense(4, 2), 0.4),
                                    (:B, :to, :A) => GINConv(Dense(4, 2), 0.4));
        y = layers(hg, x); 
        @test size(y.A) == (2, 2) && size(y.B) == (2, 3)



begin
    using MLDatasets
    using GraphNeuralNetworks
    using Flux
    using Flux: onecold, onehotbatch, logitcrossentropy
    using Plots
    #using PlutoUI
    using TSne
    using Random
    using Statistics

    ENV["DATADEPS_ALWAYS_ACCEPT"] = "true"
    Random.seed!(17) # for reproducibility
end;        


function visualize_tsne(out, targets)
        z = tsne(out, 2)
        scatter(z[:, 1], z[:, 2], color = Int.(targets[1:size(z, 1)]), leg = false)
end

dataset = Cora()

dataset.metadata

dataset.graphs

g = mldataset2gnngraph(dataset)


    # Gather some statistics about the graph.
    println("Number of nodes: $(g.num_nodes)")
    println("Number of edges: $(g.num_edges)")
    println("Average node degree: $(g.num_edges / g.num_nodes)")
    println("Number of training nodes: $(sum(g.ndata.train_mask))")
    println("Training node label rate: $(mean(g.ndata.train_mask))")
    # println("Has isolated nodes: $(has_isolated_nodes(g))")
    println("Has self-loops: $(has_self_loops(g))")
    println("Is undirected: $(is_bidirected(g))")


begin
        x = g.ndata.features
        # we onehot encode both the node labels (what we want to predict):
        y = onehotbatch(g.ndata.targets, 1:7)
        train_mask = g.ndata.train_mask
        num_features = size(x)[1]
        hidden_channels = 16
        num_classes = dataset.metadata["num_classes"]
end;


begin
    struct GCN
        layers::NamedTuple
    end

    Flux.@functor GCN # provides parameter collection, gpu movement and more

    function GCN(num_features, num_classes, hidden_channels; drop_rate = 0.5)
        layers = (conv1 = GCNConv(num_features => hidden_channels),
                  drop = Dropout(drop_rate),
                  conv2 = GCNConv(hidden_channels => num_classes))
        return GCN(layers)
    end

    function (gcn::GCN)(g::GNNGraph, x::AbstractMatrix)
        l = gcn.layers
        x = l.conv1(g, x)
        x = relu.(x)
        x = l.drop(x)
        x = l.conv2(g, x)
        return x
    end
end

begin
    gcn = GCN(num_features, num_classes, hidden_channels)
    h_untrained = gcn(g, x) |> transpose
    visualize_tsne(h_untrained, g.ndata.targets)
end


function train(model::GCN, g::GNNGraph, x::AbstractMatrix, epochs::Int, opt)
    Flux.trainmode!(model)

    for epoch in 1:epochs
        loss, grad = Flux.withgradient(model) do model
            ŷ = model(g, x)
            logitcrossentropy(ŷ[:, train_mask], y[:, train_mask])
        end

        Flux.update!(opt, model, grad[1])
        if epoch % 200 == 0
            @show epoch, loss
        end
    end
end


function accuracy(model::GCN, g::GNNGraph, x::AbstractMatrix, y::Flux.OneHotArray,
                  mask::BitVector)
    Flux.testmode!(model)
    mean(onecold(model(g, x))[mask] .== onecold(y)[mask])
end

begin
    opt_gcn = Flux.setup(Adam(1e-2), gcn)
    train(gcn, g, x, epochs, opt_gcn)
end


train_accuracy = accuracy(gcn, g, g.ndata.features, y, train_mask)
test_accuracy = accuracy(gcn, g, g.ndata.features, y, .!train_mask)

println("Train accuracy: $(train_accuracy)")
println("Test accuracy: $(test_accuracy)")





function remove_edges(g::GNNGraph, edges_to_remove::Vector{Int})
    s, t = GraphNeuralNetworks.edge_index(g)
    w = GraphNeuralNetworks.get_edge_weight(g)
    edata = g.edata

    mask_to_keep = .!in.(1:g.num_edges, edges_to_remove)
    s = s[mask_to_keep]
    t = t[mask_to_keep]
    edata = GraphNeuralNetworks.getobs(edata, mask_to_keep)
    w = isnothing(w) ? nothing : GraphNeuralNetworks.getobs(w, mask_to_keep)

    GNNGraph((s, t, w),
             g.num_nodes, length(s), g.num_graphs,
             g.graph_indicator,
             g.ndata, edata, g.gdata)
end





using GraphNeuralNetworks, Graphs, SparseArrays


# Construct a GNNGraph from from a Graphs.jl's graph
lg = erdos_renyi(10, 30)
g = GNNGraph(lg)

# Same as above using convenience method rand_graph
g = rand_graph(10, 60)

# From an adjacency matrix
A = sprand(10, 10, 0.3)
g = GNNGraph(A)

# From an adjacency list
adjlist = [[2,3], [1,3], [1,2,4], [3]]
g = GNNGraph(adjlist)

# From COO representation
source = [1,1,2,2,3,3,3,4,2]
target = [2,3,1,3,1,2,4,3,2]
g = GNNGraph(source, target)

has_self_loops(g)

g = remove_self_loops(g)

has_self_loops(g)
@assert g.num_edges == 8 
g.num_edges


s = [1, 1, 2, 3]
        t = [2, 3, 4, 5]
        g = GNNGraph(s, t)
        snew = [1, 3]
        tnew = [4, 3]
        gnew = add_edges(g, snew, tnew)
        @test gnew.num_edges == 6
        @test sort(inneighbors(gnew, 4)) == [1, 2]

        gnew = remove_edges(g, [6])
        @test gnew.num_edges == 5
        @test sort(inneighbors(gnew, 4)) == [1, 2]

"""
    remove_edges(g::GNNGraph, edges_to_remove::Vector{Int})

Remove specified edges from a GNNGraph.

# Arguments
- `g`: The input graph from which edges will be removed.
- `edges_to_remove`: Vector of edge indices to be removed.

# Returns
A new GNNGraph with the specified edges removed.

# Example
```julia
using GraphNeuralNetworks

# Construct a GNNGraph
g = GNNGraph([1, 1, 2, 2, 3], [2, 3, 1, 3, 1])

# Remove the second edge
g_new = remove_edges(g, [2])

println(g_new)
```
"""

function remove_edge(g::GNNGraph, edges_to_remove)
    s, t = edge_index(g)
    w = get_edge_weight(g)
    edata = g.edata

    mask_to_keep = trues(length(s))

    for edge_index in edges_to_remove
        mask_to_keep[edge_index] = false
    end

    s = s[mask_to_keep]
    t = t[mask_to_keep]
    edata = getobs(edata, mask_to_keep)
    println(edata)
    w = isnothing(w) ? nothing : getobs(w, mask_to_keep)

    GNNGraph((s, t, w),
             g.num_nodes, length(s), g.num_graphs,
             g.graph_indicator,
             g.ndata, edata, g.gdata)
end


        s = [1, 1, 2, 3]
        t = [2, 3, 4, 5]
        g = GNNGraph(s, t)
        snew = [1, 3]
        tnew = [4, 3]
        gnew = add_edges(g, snew, tnew)
        @test gnew.num_edges == 6
        @test sort(inneighbors(gnew, 4)) == [1, 2]

        gnew = remove_edge(gnew, [1])
        @test gnew.num_edges == 5
        @test sort(inneighbors(gnew, 4)) == [1, 2]


        using Test
        s = [1, 1, 2, 3]
        t = [2, 3, 4, 5]
        g = GNNGraph(s, t)    
        gnew = remove_edges(g, [1])
        new_s, new_t = edge_index(gnew)
        @test gnew.num_edges == 3
        @test new_s == s[2:end]
        @test new_t == t[2:end]


using GraphNeuralNetworks, Graphs, SparseArrays

# Construct a GNNGraph from COO representation
source = [1, 1, 2, 2, 3, 3, 3, 4, 2,3]
target = [2, 3, 1, 3, 1, 2, 4, 3, 2,3]
g = GNNGraph(source, target)

# Display the initial number of edges
println("Initial number of edges: $(g.num_edges)")

# Define the edges to be removed
edges_to_remove = [1, 2, 4]  # Example: removing the first, second, and fourth edges

# Remove the specified edges
g_new = remove_edge(g, edges_to_remove)

# Display the number of edges after removal
println("Number of edges after removal: $(g_new.num_edges)")

# Check if the new graph has self-loops after removal
println("Does the new graph have self-loops? $(has_self_loops(g_new))")

s = [1, 1, 2, 3]
        t = [2, 3, 4, 5]
        g = GNNGraph(s, t, graph_type = GRAPH_T)
        
        gnew = remove_edges(g, [1])
        @test gnew.num_edges == 3



begin
    struct MLP
        layers::NamedTuple
    end

    Flux.@functor MLP

    function MLP(num_features, num_classes, hidden_channels; drop_rate = 0.5)
        layers = (hidden = Dense(num_features => hidden_channels),
                  drop = Dropout(drop_rate),
                  classifier = Dense(hidden_channels => num_classes))
        return MLP(layers)
    end

    function (model::MLP)(x::AbstractMatrix)
        l = model.layers
        x = l.hidden(x)
        x = relu(x)
        x = l.drop(x)
        x = l.classifier(x)
        return x
    end
end

function train(model::MLP, data::AbstractMatrix, epochs::Int, opt)
    Flux.trainmode!(model)

    for epoch in 1:epochs
        loss, grad = Flux.withgradient(model) do model
            ŷ = model(data)
            logitcrossentropy(ŷ[:, train_mask], y[:, train_mask])
        end

        Flux.update!(opt, model, grad[1])
        if epoch % 200 == 0
            @show epoch, loss
        end
    end
end

function accuracy(model::MLP, x::AbstractMatrix, y::Flux.OneHotArray, mask::BitVector)
    Flux.testmode!(model)
    mean(onecold(model(x))[mask] .== onecold(y)[mask])
end

begin
    mlp = MLP(num_features, num_classes, hidden_channels)
    opt_mlp = Flux.setup(Adam(1e-3), mlp)
    epochs = 20
    train(mlp, g.ndata.features, epochs, opt_mlp)
end

accuracy(mlp, g.ndata.features, y, .!train_mask)




import Pkg;
Pkg.add("MLDatasets")
Pkg.add("Plots")
Pkg.add("TSne")


using MLDatasets
using GraphNeuralNetworks
using Flux
using Flux: onecold, onehotbatch, logitcrossentropy
using Plots
using TSne
using Random
using Statistics

ENV["DATADEPS_ALWAYS_ACCEPT"] = "true"
Random.seed!(17) # for reproducibility

function visualize_tsne(out, targets)
    z = tsne(out, 2)
    scatter(z[:, 1], z[:, 2], color = Int.(targets[1:size(z, 1)]), leg = false)
end

# Function to remove specific edges from a GNNGraph
function remove_edg(g::GNNGraph, edges_to_remove::Vector{Int})
    s, t = GraphNeuralNetworks.edge_index(g)
    w = GraphNeuralNetworks.get_edge_weight(g)
    edata = g.edata

    mask_to_keep = .!in.(1:g.num_edges, edges_to_remove)
    s = s[mask_to_keep]
    t = t[mask_to_keep]
    edata = GraphNeuralNetworks.getobs(edata, mask_to_keep)
    w = isnothing(w) ? nothing : GraphNeuralNetworks.getobs(w, mask_to_keep)

    return GNNGraph((s, t, w),
                    g.num_nodes, length(s), g.num_graphs,
                    g.graph_indicator,
                    g.ndata, edata, g.gdata)
end

dataset = Cora()

dataset.metadata

dataset.graphs

g = mldataset2gnngraph(dataset)

# Gather some statistics about the graph.
println("Number of nodes: $(g.num_nodes)")
println("Number of edges: $(g.num_edges)")
println("Average node degree: $(g.num_edges / g.num_nodes)")
println("Number of training nodes: $(sum(g.ndata.train_mask))")
println("Training node label rate: $(mean(g.ndata.train_mask))")
# println("Has isolated nodes: $(has_isolated_nodes(g))")
println("Has self-loops: $(has_self_loops(g))")
println("Is undirected: $(is_bidirected(g))")

x = g.ndata.features
# we onehot encode both the node labels (what we want to predict):
y = onehotbatch(g.ndata.targets, 1:7)
train_mask = g.ndata.train_mask
num_features = size(x)[1]
hidden_channels = 16
num_classes = dataset.metadata["num_classes"]

struct GCN
    layers::NamedTuple
end

Flux.@functor GCN # provides parameter collection, gpu movement and more

function GCN(num_features, num_classes, hidden_channels; drop_rate = 0.5)
    layers = (conv1 = GCNConv(num_features => hidden_channels),
              drop = Dropout(drop_rate),
              conv2 = GCNConv(hidden_channels => num_classes))
    return GCN(layers)
end

function (gcn::GCN)(g::GNNGraph, x::AbstractMatrix)
    l = gcn.layers
    x = l.conv1(g, x)
    x = relu.(x)
    x = l.drop(x)
    x = l.conv2(g, x)
    return x
end

function remove_edgess(g::GNNGraph, edges_to_remove)
    s, t = edge_index(g)
    w = get_edge_weight(g)
    edata = g.edata

    mask_to_keep = trues(length(s))

    mask_to_keep[edges_to_remove] .= false

    s = s[mask_to_keep]
    t = t[mask_to_keep]
    edata = getobs(edata, mask_to_keep)
    w = isnothing(w) ? nothing : getobs(w, mask_to_keep)

    GNNGraph((s, t, w),
             g.num_nodes, length(s), g.num_graphs,
             g.graph_indicator,
             g.ndata, edata, g.gdata)
end



function train(model::GCN, g::GNNGraph, x::AbstractMatrix, epochs::Int, opt, p_drop::Float64)
    Flux.trainmode!(model)

    for epoch in 1:epochs
        # Generate a list of edges to remove based on the probability
        #num_edges = g.num_edges
        #edges_to_remove = filter(_ -> rand() < p_drop, 1:num_edges)
        
        # Remove the selected edges
        #g_dropped = remove_edgess(g, edges_to_remove)
        #print(g_dropped)
        g_dropped = g
        g_dropped = drop_edge(g_dropped, Float32(1.0))
        
        loss, grad = Flux.withgradient(model) do model
            ŷ = model(g_dropped, x)
            logitcrossentropy(ŷ[:, train_mask], y[:, train_mask])
        end

        Flux.update!(opt, model, grad[1])
        if epoch % 200 == 0
            @show epoch, loss
        end
    end
end

function drop_edge(g::GNNGraph, p::Float32 = Float32(0.5))
    num_edges = g.num_edges
    edges_to_remove = filter(_ -> rand() < p, 1:num_edges)        
    g = remove_edgess(g, edges_to_remove)
    s, t = edge_index(g)
    w = get_edge_weight(g)
    edata = g.edata
    GNNGraph((s, t, w),
             g.num_nodes, length(s), g.num_graphs,
             g.graph_indicator,
             g.ndata, edata, g.gdata)
end

g_dropped = g
print(g_dropped.num_edges)
g_dropped = drop_edge(g_dropped, Float32(1.0))
print(g_dropped.num_edges)

function accuracy(model::GCN, g::GNNGraph, x::AbstractMatrix, y::Flux.OneHotArray,
                  mask::BitVector)
    Flux.testmode!(model)
    mean(onecold(model(g, x))[mask] .== onecold(y)[mask])
end

gcn = GCN(num_features, num_classes, hidden_channels)
epochs = 20
p_drop = 0.1 # Probability of dropping an edge

opt_gcn = Flux.setup(Adam(1e-2), gcn)
train(gcn, g, x, epochs, opt_gcn, p_drop)

train_accuracy = accuracy(gcn, g, g.ndata.features, y, train_mask)
test_accuracy = accuracy(gcn, g, g.ndata.features, y, .!train_mask)

println("Train accuracy: $(train_accuracy)")
println("Test accuracy: $(test_accuracy)")

hg = rand_bipartite_heterograph((2,3), 6)
x = (A = rand(Float32, 2, 2), B = rand(Float32, 3, 3))

layers = HeteroGraphConv((:A, :to, :B) => AGNNConv(init_beta=1.0, add_self_loops=true, trainable=true),
                         (:B, :to, :A) => AGNNConv(init_beta=1.0, add_self_loops=true, trainable=true));

