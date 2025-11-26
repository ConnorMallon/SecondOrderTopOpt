#############################################################################
# ṗ->u̇ : # Solving the "incremental state equation" ∂R/∂u * u̇ = - ∂R/∂p * ṗ #
#############################################################################

function incremental_adjoint_partials(res,uh,ph,λh,spaces)
  U,V,V_p = spaces

  # ∂²R / ∂u² * u̇ * λ
  ∂2R∂u2 = Gridap.hessian(uh->res(uh,λh,ph),uh) 
  ∂2R∂u2_mat = assemble_matrix(∂2R∂u2,U,V)  

  # ∂/∂p (∂R/∂u * λ) * ṗ
  ∂R∂u_λ(uh,ph) = Gridap.gradient(uh->res(uh,λh,ph),uh)
  ∂2R∂u∂p = Gridap.jacobian(p->∂R∂u_λ(uh,p),ph) 
  ∂2R∂u∂p_mat = assemble_matrix(∂2R∂u∂p,V_p,V)

  # ∂²R / ∂p² * ṗ * λ
  ∂2R∂p2 = Gridap.hessian(ph->res(uh,λh,ph),ph)
  ∂2R∂p2_mat = assemble_matrix(∂2R∂p2,V_p,V_p)

  # ∂/∂u (∂R/∂p * λ) * ṗ
  ∂R∂p_λ(uh,ph) = Gridap.gradient(ph->res(uh,λh,ph),ph)
  ∂2R∂p∂u = Gridap.jacobian(uh->∂R∂p_λ(uh,ph),uh) 
  ∂2R∂p∂u_mat = assemble_matrix(∂2R∂p∂u,U,V_p)

  return ∂2R∂u2_mat, ∂2R∂u∂p_mat, ∂2R∂p2_mat, ∂2R∂p∂u_mat
end

function (p_to_u::NonlinearFEStateMap)(pᵋ::Vector{ForwardDiff.Dual{T,VT,PT}}) where {T,VT,PT}
  U,V,V_p = p_to_u.spaces
  res = p_to_u.res
  
  p = ForwardDiff.value.(pᵋ)
  ph = FEFunction(V_p,p)
  ṗ =  mapreduce(ForwardDiff.partials, vcat, pᵋ)'

  # pushforward the value 
  u = p_to_u(p)
  uh = FEFunction(U,u) 

  # pushforward the dual component (solve the incremental state equation)
  # this should be retrieved from whatever matrix is used in the forward pass -- how can it be done cleanly so we dont have to write seperate methods for Affine and Nonlinear maps
  dv = get_fe_basis(V)
  ∂R∂u = Gridap.jacobian(uh->res(uh,dv,ph),uh) 
  ∂R∂u_mat = assemble_matrix(∂R∂u,U,V)  

  # once per outer iteration
  dv = get_fe_basis(V)
  ∂R∂p = Gridap.jacobian(p->res(uh,dv,p),ph)
  ∂R∂p_mat = assemble_matrix(∂R∂p,V_p,V)
  
  # once per inner iteration
  u̇ = ∂R∂u_mat \ (-∂R∂p_mat * ṗ')

  return map(u, eachrow(u̇)) do v, p
    ForwardDiff.Dual{T}(v, p...)
  end
end

function ChainRulesCore.rrule(p_to_u::NonlinearFEStateMap,pᵋ::Vector{ForwardDiff.Dual{T,VT,PT}}) where {T,VT,PT}
  spaces = p_to_u.spaces
  U,V,V_p = spaces
  res = p_to_u.res
  adjoint_ns, _, λ = p_to_u.cache.adj_cache

  uᵋ = p_to_u(pᵋ)
  p = ForwardDiff.value.(pᵋ)
  ph = FEFunction(V_p,p)
  ṗ =  vec(mapreduce(ForwardDiff.partials, hcat, pᵋ))
  u = ForwardDiff.value.(uᵋ)
  uh = FEFunction(U,u)
  u̇ = vec(mapreduce(ForwardDiff.partials, hcat, uᵋ))

  function p_to_u_pullback(duᵋ)
    # pullback the value 
    du = ForwardDiff.value.(duᵋ)
    dudp_vec, assem_deriv = get_plb_cache(p_to_u)
    λ =  solve!(λ,adjoint_ns,du)
    λh = FEFunction(V,λ)
    ∂R∂p_λ = Gridap.gradient(ph->res(uh,λh,ph),ph)
    ∂R∂p_vec_λ = assemble_vector(∂R∂p_λ,V_p)
    dp = - ∂R∂p_vec_λ

    # pullback the dual component
    du̇ = vec(mapreduce(ForwardDiff.partials, hcat, duᵋ))

    # new caches - needs work
    λ⁻ = copy(λ)

    # once per outer iteration
    ∂2R∂u2_mat, ∂2R∂u∂p_mat, ∂2R∂p2_mat, ∂2R∂p∂u_mat = incremental_adjoint_partials(res,uh,ph,λh,spaces)

    # once per inner iteration
    du̇_R = ∂2R∂u2_mat*u̇ + ∂2R∂u∂p_mat*ṗ
    dṗ_R = ∂2R∂p2_mat*ṗ + ∂2R∂p∂u_mat*u̇
    λ⁻ = solve!(λ⁻,adjoint_ns,du̇-du̇_R)
    λ⁻h = FEFunction(V,λ⁻)
    ∂R∂p_λ⁻ = Gridap.gradient(ph->res(uh,λ⁻h,ph),ph)
    ∂R∂p_vec_λ⁻ = assemble_vector(∂R∂p_λ⁻,V_p)
    dṗ_adj = - ∂R∂p_vec_λ⁻ - dṗ_R

    dpᵋ = map(dp, eachrow(dṗ_adj)) do v, p
      ForwardDiff.Dual{T}(v, p...)
    end
    ( NoTangent(), dpᵋ)
  end

  return uᵋ, p_to_u_pullback
end

######################################################################
# u̇ -> du̇, dṗ: Computing the increments of the objective functional #
######################################################################

function incremental_objective_partials(F,uh,ph,spaces)
  U,V_p = spaces
  
  # ∂²J / ∂u² * u̇
  ∂2J∂u2 = Gridap.hessian(uh->F(uh,ph),uh)
  ∂2J∂u2_mat = assemble_matrix(∂2J∂u2,U,U)

  # ∂/∂p (∂J/∂u ) * ṗ
  ∂J∂u(uh,ph) = Gridap.gradient(uh->F(uh,ph),uh)
  ∂2J∂u∂p = Gridap.jacobian(p->∂J∂u(uh,p),ph)
  ∂2J∂u∂p_mat = assemble_matrix(∂2J∂u∂p,V_p,U)

  # ∂²J / ∂p² * ṗ
  ∂2J∂p2 = Gridap.hessian(p->F(uh,p),ph)
  ∂2J∂p2_mat = assemble_matrix(∂2J∂p2,V_p,V_p)

  # ∂/∂u (∂J / ∂p) * u̇
  ∂J∂p(uh,ph) = Gridap.gradient(p->F(uh,p),ph)
  ∂2J∂p∂u = Gridap.jacobian(uh->∂J∂p(uh,ph),uh)
  ∂2J∂p∂u_mat = assemble_matrix(∂2J∂p∂u,U,V_p)

  return ∂2J∂u2_mat, ∂2J∂u∂p_mat, ∂2J∂p2_mat, ∂2J∂p∂u_mat
end

function (u_to_j::StateParamMap)(uᵋ::Vector{ForwardDiff.Dual{T1,V1,P1}},pᵋ::Vector{ForwardDiff.Dual{T2,V2,P2}}) where {T1,V1,P1,T2,V2,P2}
  F = u_to_j.F
  U,V_p = u_to_j.spaces

  # pushforward the value # skip if already computed at the point p 
  uh = FEFunction(U,ForwardDiff.value.(uᵋ))
  ph = FEFunction(V_p,ForwardDiff.value.(pᵋ))
  J = sum(F(uh,ph))

  # pushforward the dual component 
  u̇ = ForwardDiff.partials.(uᵋ)
  ṗ = ForwardDiff.partials.(pᵋ)
  ∂F∂u = Gridap.gradient(uh->F(uh,ph),uh) 
  ∂F∂u_vec = assemble_vector(∂F∂u,U)
  ∂F∂p = Gridap.gradient(ph->F(uh,ph),ph)
  ∂F∂p_vec = assemble_vector(∂F∂p,V_p)
  J̇ = ∂F∂p_vec ⋅ ṗ + ∂F∂u_vec ⋅ u̇

  Jᵋ = ForwardDiff.Dual{T2}(J, J̇)
  return  Jᵋ
end

function ChainRulesCore.rrule(u_to_j::StateParamMap,uᵋ::Vector{ForwardDiff.Dual{T1,V1,P1}},pᵋ::Vector{ForwardDiff.Dual{T2,V2,P2}}) where {T1,V1,P1,T2,V2,P2}
  spaces = u_to_j.spaces
  U,V_p = spaces
  F = u_to_j.F

  uh = FEFunction(U,ForwardDiff.value.(uᵋ))
  ph = FEFunction(V_p,ForwardDiff.value.(pᵋ))

  function u_to_j_pullback(dJᵋ)
    # pullback the value # skip if already computed at the point p
    dJ = ForwardDiff.value(dJᵋ)
    ∂F∂u = Gridap.gradient(uh->F(uh,ph),uh) 
    ∂F∂u_vec = assemble_vector(∂F∂u,U) 
    ∂F∂p = Gridap.gradient(ph->F(uh,ph),ph)
    ∂F∂p_vec = assemble_vector(∂F∂p,V_p)
    du = dJ * ∂F∂u_vec
    dp = dJ * ∂F∂p_vec

    # pullback the dual component

    # once per outer iteration
    ∂2J∂u2_mat, ∂2J∂u∂p_mat, ∂2J∂p2_mat, ∂2J∂p∂u_mat = incremental_objective_partials(F,uh,ph,spaces)
    
    # once per inner iteration
    u̇ = mapreduce(ForwardDiff.partials, hcat, uᵋ)'
    ṗ = mapreduce(ForwardDiff.partials, hcat, pᵋ)'
    dṗ = ∂2J∂p2_mat * ṗ + ∂2J∂p∂u_mat * u̇ 
    du̇ = ∂2J∂u2_mat * u̇ + ∂2J∂u∂p_mat * ṗ 

    Du̇ = map(du, eachrow(du̇)) do v, p
      ForwardDiff.Dual{T1}(v, p...)
    end
    Dṗ = map(dp, eachrow(dṗ)) do v, p
      ForwardDiff.Dual{T2}(v, p...)
    end
    (  NoTangent(), Du̇, Dṗ )
  end

  return u_to_j(uᵋ,pᵋ), u_to_j_pullback
end

