#############################################################################
# ṗ->u̇ : # Solving the "incremental state equation" ∂R/∂u * u̇ = - ∂R/∂p * ṗ #
#############################################################################

function incremental_state_pushforward(res,uh,φh,ṗ,spaces)
    U,V,V_φ = spaces 

    # RHS: 
    dv = get_fe_basis(V)
    ∂R∂φ = Gridap.jacobian(φ->res(uh,dv,φ),φh)
    ∂R∂φ_mat = assemble_matrix(∂R∂φ,V_φ,V)
    ∂R∂u_mat_ṗ = ∂R∂φ_mat * ṗ

    # LHS:
    ∂R∂u = Gridap.jacobian(uh->res(uh,dv,φh),uh) 
    ∂R∂u_mat = assemble_matrix(∂R∂u,U,V)  

    # solving  
    u̇ = ∂R∂u_mat \ (-∂R∂u_mat_ṗ)
end 

######################################################################
# u̇ -> du̇, dṗ: Computing the increments of the objective functional #
######################################################################

function incremental_objective_partials(J,uh,φh,spaces)
    U,V,V_φ = spaces
        # ∂²J / ∂u² * u̇
    ∂2J∂u2 = Gridap.hessian(uh->J(uh,φh),uh)
    ∂2J∂u2_mat = assemble_matrix(∂2J∂u2,V,V)

    # ∂/∂p (∂J/∂u ) * ṗ
    ∂J∂u(uh,φh) = Gridap.gradient(uh->J(uh,φh),uh)
    ∂2J∂u∂φ = Gridap.jacobian(φ->∂J∂u(uh,φ),φh)
    ∂2J∂u∂φ_mat = assemble_matrix(∂2J∂u∂φ,V_φ,V)

    # ∂²J / ∂p² * ṗ
    ∂2J∂φ2 = Gridap.hessian(φ->J(uh,φ),φh)
    ∂2J∂φ2_mat = assemble_matrix(∂2J∂φ2,V_φ,V_φ)

    # ∂/∂u (∂J / ∂p) * u̇
    ∂J∂φ(uh,φh) = Gridap.gradient(φ->J(uh,φ),φh)
    ∂2J∂φ∂u = Gridap.jacobian(uh->∂J∂φ(uh,φh),uh)
    ∂2J∂φ∂u_mat = assemble_matrix(∂2J∂φ∂u,U,V_φ)

    return ∂2J∂u2_mat, ∂2J∂u∂φ_mat, ∂2J∂φ2_mat, ∂2J∂φ∂u_mat
end

function incremental_objective_pushforward(J,uh,φh,u̇,ṗ,spaces)

    ∂2J∂u2_mat, ∂2J∂u∂φ_mat, ∂2J∂φ2_mat, ∂2J∂φ∂u_mat = incremental_objective_partials(J,uh,φh,spaces)

    dṗ = ∂2J∂φ2_mat*ṗ + ∂2J∂φ∂u_mat*u̇
    du̇ = ∂2J∂u2_mat*u̇ + ∂2J∂u∂φ_mat*ṗ

    return du̇, dṗ
end

################################################################################################################
# du̇->dṗ : Solving the "incremental adjoint equation" ∂R/∂uᵗ * λ⁻ = du̇ - ∂²R/∂u² * u̇ * λ - ∂/∂p(∂R/∂u) * ṗ * λ #
################################################################################################################  

function incremental_adjoint_partials(res,u,φ,λ,spaces)
    U,V,V_φ = spaces

    uh = FEFunction(U,u)
    φh = FEFunction(V_φ,φ)
    λh = FEFunction(V,λ)

    # ∂²R / ∂u² * u̇ * λ
    ∂2R∂u2 = Gridap.hessian(uh->res(uh,λh,φh),uh) 
    ∂2R∂u2_mat = assemble_matrix(∂2R∂u2,U,V)  
    #∂2R∂u2_mat_u̇ = ∂2R∂u2_mat * u̇

    # ∂/∂p (∂R/∂u * λ) * ṗ
    ∂R∂u_λ(uh,φh) = Gridap.gradient(uh->res(uh,λh,φh),uh)
    ∂2R∂u∂φ = Gridap.jacobian(φ->∂R∂u_λ(uh,φ),φh) 
    ∂2R∂u∂φ_mat = assemble_matrix(∂2R∂u∂φ,V_φ,V)
    #∂2R∂u∂φ_mat_ṗ = ∂2R∂u∂φ_mat * ṗ

    # ∂²R / ∂p² * ṗ * λ
    ∂2R∂φ2 = Gridap.hessian(φh->res(uh,λh,φh),φh)
    ∂2R∂φ2_mat = assemble_matrix(∂2R∂φ2,V_φ,V_φ)
    #∂2R∂φ2_mat_ṗ = ∂2R∂φ2_mat * ṗ

    # ∂/∂u (∂R/∂p * λ) * ṗ
    ∂R∂φ_λ(uh,φh) = Gridap.gradient(φh->res(uh,λh,φh),φh)
    ∂2R∂φ∂u = Gridap.jacobian(uh->∂R∂φ_λ(uh,φh),uh) 
    ∂2R∂φ∂u_mat = assemble_matrix(∂2R∂φ∂u,U,V_φ)
    #∂2R∂φ∂u_mat_u̇ = ∂2R∂φ∂u_mat * u̇

    return ∂2R∂u2_mat, ∂2R∂u∂φ_mat, ∂2R∂φ2_mat, ∂2R∂φ∂u_mat
end

function incremental_adjoint_pushforward(φ_to_u,u̇,ṗ,du̇)
    spaces = φ_to_u.spaces
    _,V,V_φ = spaces
    adjoint_ns, _, λ = φ_to_u.cache.adj_cache
    res = φ_to_u.res
    uh = get_state(φ_to_u)
    u = uh.free_values
    φ = get_parameter(φ_to_u)

    φh = FEFunction(V_φ,φ)

    # once per outer iteration
    ∂2R∂u2_mat, ∂2R∂u∂φ_mat, ∂2R∂φ2_mat, ∂2R∂φ∂u_mat = incremental_adjoint_partials(res,u,φ,λ,spaces)

    # once per inner iteration
    du̇_R = ∂2R∂u2_mat*u̇ + ∂2R∂u∂φ_mat*ṗ
    dφ̇_R = ∂2R∂φ2_mat*ṗ + ∂2R∂φ∂u_mat*u̇

    λ⁻ = copy(λ)
    λ⁻ = solve!(λ⁻,adjoint_ns,du̇-du̇_R)
    λ⁻h = FEFunction(V,λ⁻)
    ∂R∂p_λ⁻ = Gridap.gradient(φh->res(uh,λ⁻h,φh),φh)
    ∂R∂p_mat_λ⁻ = assemble_vector(∂R∂p_λ⁻,V_φ)
    dṗ_adj = - ∂R∂p_mat_λ⁻ - dφ̇_R
end