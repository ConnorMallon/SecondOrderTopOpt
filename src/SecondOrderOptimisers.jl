
function NewtonCG(pcfs::CustomPDEConstrainedFunctionals,p0)
  
  p_to_j = pcfs.φ_to_jc
  ∇f = p->Zygote.gradient(p->p_to_j(p)[1],p)[1]
  Hṗ(p,ṗ) =  ForwardDiff.derivative(α -> ∇f(p + α*ṗ), 0)
  function f(x::Vector)
    p_to_j(x)[1]
  end
  function fg!(G,x)
    F,Gs = Zygote.withgradient(p->p_to_j(p)[1], x)
    copyto!(G, Gs[1])
    F[1]
  end
  function hv!(Hv, x, v)
    copyto!(Hv, Hṗ(x,v))
    Hv
  end
  d = Optim.TwiceDifferentiableHV(f,fg!,hv!,p0)
  result = Optim.optimize(d, p0, Optim.KrylovTrustRegion(),
              Optim.Options(g_tol = 1e-12,
                            iterations = 10,
                            show_trace = true,
                            store_trace = true,
              ))
end