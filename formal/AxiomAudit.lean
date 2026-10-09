import ProMax
import SequenceDilution
import ErrorDecomp
import CoarseGrain
import KLChain
import SequenceTV
import KLEquality
import CoarseKL
import K3Inlier
import SparseReturn
import DiscountedGradient
import TreePinsker
import AdaptiveBound
import TreePrefixGate
import PrefixGateLimits
import PrefixGateMoments
import PrefixGateExample
import SelectiveRewardExample
import SelectiveRolloutBound
import CoupledGateCounterexample
import PositiveCertificateExample
import RLOOProbability
import RLOOGeneral
import RLOOMoments
import RLOOScaling
import PolicyGradient
import PromptGradient
import TreePolicyGradient
import TokenScoreBridge
import BatchReduction
import BatchPerformance
import FiniteSelection
import ProMaxCertificate
import EOSCertificateBridge
import CachedGateDrift
import EOSGateDrift
import UniformEligibility
import EmpiricalCertificate
import GroupCertificate
import PolicyCover
import RLOOSurrogate
import SurrogateBridge
import ProMaxObjective
import TwoPolicyObjective
import PPODerivative
import AutoregressiveTree
import TreeSurrogate
import PromptMixture
import PrefixConcentration
import ConcentrationPromptMixture
import PrefixMasking
import TokenMaskContext
import StructuralExamples
import GroupFiltering
import ImplementationBridge
import PaddedPrefix
import PrefixDefinition
import NormalizationSafeguards
import AdaptiveMechanism
import LogitBounds
import UniformScale
import BinaryUpdate
import BinaryLength
import BinaryAscent
import PromptAscent
import BatchDenominator
import TreeKL
import TreeTV

/-!
Audit every public theorem in the project namespace, including transitive
dependencies of its type and proof. A missing proof (`sorryAx`), a new custom
axiom, or an axiom introduced by a computational shortcut fails the build.
This check supplements kernel checking; it does not validate a theorem's
mathematical modelling assumptions or its match to an English statement.
-/
run_cmd do
  let env ← Lean.getEnv
  let allowed : Array Lean.Name := #[``propext, ``Classical.choice, ``Quot.sound]
  let mut count := 0
  for (name, info) in env.constants.toList do
    if (`REINFORCEProMax).isPrefixOf name && info.isTheorem then
      count := count + 1
      let dependencies ← Lean.collectAxioms name
      for axiomName in dependencies do
        unless allowed.contains axiomName do
          throwError "Unapproved axiom {axiomName} in theorem {name}"
  if count == 0 then
    throwError "No project theorems found; the axiom audit did not run."
  Lean.logInfo m!"Axiom audit passed for {count} project theorems; only propext, Classical.choice, and Quot.sound are allowed."

-- Current two-policy objective; historical clipped-objective results remain separate.
#print axioms REINFORCEProMax.TwoPolicy.candidate_gate_residual
#print axioms REINFORCEProMax.TwoPolicy.candidate_gate_residual_bound
#print axioms REINFORCEProMax.TwoPolicy.objective_split
#print axioms REINFORCEProMax.TwoPolicy.scaled_raw_lower_bound
