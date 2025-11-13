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

function objective_partials(J,uh,φh,u̇,ṗ,spaces )
    U,V,V_φ = spaces
        # ∂²J / ∂u² * u̇
    ∂2J∂u2 = Gridap.hessian(uh->J(uh,φh),uh)
    ∂2J∂u2_mat = assemble_matrix(∂2J∂u2,V,V)
    ∂2J∂u2_mat_u̇ = ∂2J∂u2_mat * u̇

    # ∂/∂p (∂J/∂u ) * ṗ
    ∂J∂u(uh,φh) = Gridap.gradient(uh->J(uh,φh),uh)
    ∂2J∂u∂φ = Gridap.jacobian(φ->∂J∂u(uh,φ),φh)
    ∂2J∂u∂φ_mat = assemble_matrix(∂2J∂u∂φ,V_φ,V)
    ∂2J∂u∂φ_mat_ṗ = ∂2J∂u∂φ_mat * ṗ

    # ∂²J / ∂p² * ṗ
    ∂2J∂φ2 = Gridap.hessian(φ->J(uh,φ),φh)
    ∂2J∂φ2_mat = assemble_matrix(∂2J∂φ2,V_φ,V_φ)
    ∂2J∂φ2_mat_ṗ = ∂2J∂φ2_mat * ṗ

    # ∂/∂u (∂J / ∂p) * u̇
    ∂J∂φ(uh,φh) = Gridap.gradient(φ->J(uh,φ),φh)
    ∂2J∂φ∂u = Gridap.jacobian(uh->∂J∂φ(uh,φh),uh)
    ∂2J∂φ∂u_mat = assemble_matrix(∂2J∂φ∂u,U,V_φ)
    ∂2J∂φ∂u_mat_u̇ = ∂2J∂φ∂u_mat * u̇

    return ∂2J∂u2_mat_u̇, ∂2J∂u∂φ_mat_ṗ, ∂2J∂φ2_mat_ṗ, ∂2J∂φ∂u_mat_u̇
end

function inc_objective_pullback_pushforward(J,uh,φh,u̇,ṗ,spaces)

    ∂2J∂u2_mat_u̇, ∂2J∂u∂φ_mat_ṗ, ∂2J∂φ2_mat_ṗ, ∂2J∂φ∂u_mat_u̇ = objective_partials(J,uh,φh,u̇,ṗ,spaces)

    dṗ = ∂2J∂φ2_mat_ṗ + ∂2J∂φ∂u_mat_u̇
    du̇ = ∂2J∂u2_mat_u̇ + ∂2J∂u∂φ_mat_ṗ

    return du̇, dṗ
end

function incremental_adjoint_partials(res,uh,λh,φh,u̇,ṗ,spaces)
    U,V,V_φ = spaces

    # ∂²R / ∂u² * u̇ * λ
    ∂2R∂u2 = Gridap.hessian(uh->res(uh,λh,φh),uh) 
    ∂2R∂u2_mat = assemble_matrix(∂2R∂u2,U,V)  
    ∂2R∂u2_mat_u̇ = ∂2R∂u2_mat * u̇

    # ∂/∂p (∂R/∂u * λ) * ṗ
    ∂R∂u_λ(uh,φh) = Gridap.gradient(uh->res(uh,λh,φh),uh)
    ∂2R∂u∂φ = Gridap.jacobian(φ->∂R∂u_λ(uh,φ),φh) 
    ∂2R∂u∂φ_mat = assemble_matrix(∂2R∂u∂φ,V_φ,V)
    ∂2R∂u∂φ_mat_ṗ = ∂2R∂u∂φ_mat * ṗ

    # ∂²R / ∂p² * ṗ * λ
    ∂2R∂φ2 = Gridap.hessian(φh->res(uh,λh,φh),φh)
    ∂2R∂φ2_mat = assemble_matrix(∂2R∂φ2,V_φ,V_φ)
    ∂2R∂φ2_mat_ṗ = ∂2R∂φ2_mat * ṗ

    # ∂/∂u (∂R/∂p * λ) * ṗ
    ∂R∂φ_λ(uh,φh) = Gridap.gradient(φh->res(uh,λh,φh),φh)
    ∂2R∂φ∂u = Gridap.jacobian(uh->∂R∂φ_λ(uh,φh),uh)
    ∂2R∂φ∂u_mat = assemble_matrix(∂2R∂φ∂u,U,V_φ)
    ∂2R∂φ∂u_mat_u̇ = ∂2R∂φ∂u_mat * u̇

    return ∂2R∂u2_mat_u̇, ∂2R∂u∂φ_mat_ṗ, ∂2R∂φ2_mat_ṗ, ∂2R∂φ∂u_mat_u̇

end

function incremental_adjoint_value(res,J,uh,λh,φh,u̇,ṗ,du̇,∂2R∂u2_mat_u̇,∂2R∂u∂φ_mat_ṗ,spaces)
    U,V,V_φ = spaces

    # RHS 
    inc_adjoint_rhs =  du̇ - ∂2R∂u2_mat_u̇ - ∂2R∂u∂φ_mat_ṗ

    # LHS 2w
    assem_adjoint = SparseMatrixAssembler(V,U)
    ∂R∂u_adjoint = (du,v) -> Gridap.jacobian(res,[uh,v,φh],1)
    ∂R∂u_adjoint_mat = assemble_adjoint_matrix(∂R∂u_adjoint,assem_adjoint,U,V)

    λ⁻ = ∂R∂u_adjoint_mat \ inc_adjoint_rhs

    λ⁻h = FEFunction(V,λ⁻)
end 

function incremental_adjoint_pushforward(res,J,uh,λh,φh,u̇,ṗ,du̇,spaces)

    U,V,V_φ = spaces

    ∂2R∂u2_mat_u̇, ∂2R∂u∂φ_mat_ṗ, ∂2R∂φ2_mat_ṗ, ∂2R∂φ∂u_mat_u̇ = incremental_adjoint_partials(res,uh,λh,φh,u̇,ṗ,spaces)

    λ⁻h = incremental_adjoint_value(res,J,uh,λh,φh,u̇,ṗ,du̇,∂2R∂u2_mat_u̇,∂2R∂u∂φ_mat_ṗ,spaces)

    ∂R∂p_λ⁻ = Gridap.gradient(φh->res(uh,λ⁻h,φh),φh)
    ∂R∂p_mat_λ⁻ = assemble_vector(∂R∂p_λ⁻,V_φ)

    dṗ_adj = - ∂R∂p_mat_λ⁻ - ∂2R∂φ2_mat_ṗ - ∂2R∂φ∂u_mat_u̇ 
end
