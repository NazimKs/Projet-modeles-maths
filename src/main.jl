using Random
using LinearAlgebra
using TimerOutputs
using JuMP, Cbc

# Function to generate a distance matrix for n cities
function generate_distance_matrix(n)
    points = [(rand(0:100), rand(0:100)) for i in 1:n]
    distance_matrix = zeros(Int, n, n)
    for i in 1:n
        for j in 1:n
            if i != j
                x1, y1 = points[i]
                x2, y2 = points[j]
                distance = sqrt((x2 - x1)^2 + (y2 - y1)^2)
                distance_matrix[i, j] = round(Int, distance)
            end
        end
    end
    return distance_matrix
end

# Function to generate multiple instances of different sizes
function generate_instances()
    sizes = [10, 20, 30, 50, 100]
    instances_per_size = 5
    all_instances = Dict{Int, Array{Array{Int,2},1}}()

    for size in sizes
        instances = [generate_distance_matrix(size) for _ in 1:instances_per_size]
        all_instances[size] = instances
    end
    return all_instances
end


# Functions used for constraint generation approach 

function nextCity(sol::Matrix{Int64}, start::Int64)
    n = size(sol, 1)
    for j in 1:n
        if sol[start, j] == 1
            return j
        end
    end
end

function detectSubtour(sol::Matrix{Int64})
    n = size(sol, 1)
    start = 1
    visited = [start]
    next = nextCity(sol, start)
    while next != start
        push!(visited, next)
        next = nextCity(sol, next)
    end
    push!(visited, start)
    return size(visited, 1) <= n, visited 
end

function add_subtour_elimination_constraint(model, x, visited)
    n = length(visited)
    @constraint(model, sum(x[visited[i], visited[j]] for i in 1:n-1 for j in 1:n-1) <= n - 2)
end


# Main function for the model of the constraint generation approach 

function tsp_constraint(d::Matrix{Int64}, time_limit::Int)
    n = size(d, 1)
    model = Model(Cbc.Optimizer)
    set_silent(model)
    set_optimizer_attribute(model, "seconds", time_limit)  
    @variable(model, x[i in 1:n, j in 1:n], Bin)
    @objective(model, Min, sum(d[i, j] * x[i, j] for i in 1:n, j in 1:n))
    @constraint(model, departUnique[i in 1:n], sum(x[i, j] for j in 1:n) == 1)
    @constraint(model, arriveeUnique[i in 1:n], sum(x[j, i] for j in 1:n) == 1)
    @constraint(model, sansBoucle[i in 1:n], x[i, i] == 0)
    @constraint(model, sousTour2[i in 1:n, j in 1:n; i != j], x[i, j] + x[j, i] <= 1)

    start_time = time()
    while true
        optimize!(model)
        if termination_status(model) != MOI.OPTIMAL
            println("Pas de solution optimale trouvée")
            return Inf, false, []
        end
        sol = round.(Int64, value.(x))
        existeSousTour, visited = detectSubtour(sol)
        if !existeSousTour
            arcs = [(i, j) for i in 1:n for j in 1:n if sol[i, j] == 1]
            return objective_value(model), true, arcs
        else 
            add_subtour_elimination_constraint(model, x, visited) 
        end
        elapsed_time = time() - start_time
        if elapsed_time > time_limit
            return Inf, false, []
        end
    end
end


# Main function for the model of the MTZ Formulation approach

function tsp_mtz(d::Matrix{Int64}, time_limit::Int)
    n = size(d, 1)
    model = Model(Cbc.Optimizer)
    set_silent(model)
    set_optimizer_attribute(model, "seconds", time_limit)  
    @variable(model, x[i in 1:n, j in 1:n], Bin)
    @objective(model, Min, sum(d[i, j] * x[i, j] for i in 1:n, j in 1:n))
    @constraint(model, departUnique[i in 1:n], sum(x[i, j] for j in 1:n) == 1)
    @constraint(model, arriveeUnique[i in 1:n], sum(x[j, i] for j in 1:n) == 1)
    @constraint(model, sansBoucle[i in 1:n], x[i, i] == 0)

    # additionnal constraints for MTZ Formulation

    @variable(model, 1 <= u[1:n] <= n)
    @constraint(model, u[1] == 1)
    for i in 2:n, j in 2:n
        @constraint(model, u[i] - u[j] + 1 <= (1 - x[i, j]) * (n))
    end
    
    for i in 2:n, j in 2:n
        @constraint(model, u[j] >= u[i] + 1 - (n) * (1 - x[i, j]))
    end

    start_time = time()
    while true
        optimize!(model)
        if termination_status(model) != MOI.OPTIMAL
            println("Pas de solution optimale trouvée")
            return Inf, false, []
        end
        sol = round.(Int64, value.(x))
        existeSousTour, visited = detectSubtour(sol)
        if !existeSousTour
            arcs = [(i, j) for i in 1:n for j in 1:n if sol[i, j] == 1]
            return objective_value(model), true, arcs
        end

        elapsed_time = time() - start_time
        if  elapsed_time > time_limit
            return Inf, false, []
        end
    end
end


# The main function that displays the results obtained from the 1st approach and the 2nd approach so we can compare between them

function main()
    all_instances = generate_instances()
    sizes = [10, 20, 30, 50, 100]

    for size in sizes
        for (index, matrix) in enumerate(all_instances[size])
            time_limit = 300  # 5 minutes
            println("Résolution pour l'instance $index de taille $size")
            
            println("Approche 1:")
            time_output1 = TimerOutput()
            result1 = @timeit time_output1 "TSP Time" begin
                tsp_constraint(matrix, time_limit)
            end
            distance1, optimal1, arcs1 = result1
            if optimal1
                println("Distance totale parcourue : $distance1")
                println("Les arcs utilisés sont : ")
                for (i, j) in arcs1
                    print("$i->$j   ")
                end
                println()
            else
                println("Temps limite atteint pour cette instance.")
            end
            show(time_output1)
            println("")
            
            println("Approche 2:")
            time_output2 = TimerOutput()
            result2 = @timeit time_output2 "TSP Time" begin
                tsp_mtz(matrix, time_limit)
            end
            distance2, optimal2, arcs2 = result2
            if optimal2
                println("Distance totale parcourue : $distance2")
                println("Les arcs utilisés sont : ")
                for (i, j) in arcs2
                    print("$i->$j   ")
                end
                println()
            else
                println("Temps limite atteint pour cette instance.")
            end
            show(time_output2)
            println("")
        end
    end
end

main()
