using GLPK, JuMP

# Lê a primeira instância de um arquivo TXT
function readGAP(arquivo)
    dados = readdlm(arquivo)

    m, n = Int.(dados[2,1:2])

    p = Int.(dados[ 3:(3+m-1) , 1:n])
    w = Int.(dados[ (3+m):(3+2*m-1) , 1:n])
    t = Int.(dados[ 3+2*m , 1:m ])

    return p, w, t
end

function solveGAP(p, w, t)
    GAP = Model(GLPK.Optimizer)

    # Lê dimensões a partir de p
    m, n = size(p)

    # Verifica se as dimensões dos dados são compatíveis
    if (size(a) != (m,n)) | (length(b) != m)
        error("Dados com dimensões incompatíveis!")
    end

    # insere variáveis binárias x[i,j]
    @variable(GAP, x[1:m,1:n], Bin)

    # define função objetivo
    @objective(GAP, Min, sum(c[i,j]*x[i,j] for i in 1:m, j in 1:n))

    # insere restrições de igualdade
    for j in 1:n
        @constraint(GAP, sum(x[i,j] for i in 1:m) == 1)
    end

    # insere restrições de desigualdade
    for i in 1:m
        @constraint(GAP, sum(a[i,j]*x[i,j] for j in 1:n) <= b[i])
    end

    optimize!(GAP)

    if termination_status(P) == OPTIMAL
        # retorna função objetivo e solução
        return objective_value(GAP), value.(x)
    else
        # falha: retorna FO = NaN
        return NaN, []
    end
end
