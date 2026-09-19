import ProMax
import ErrorDecomp
import CoarseGrain
import KLChain
import SequenceTV
import KLEquality
import CoarseKL
import K3Inlier
import SparseReturn
import TreePinsker
import AdaptiveBound
import TreePrefixGate
import RLOOProbability
import RLOOGeneral
import RLOOMoments
import PolicyGradient
import PromptGradient
import PromptGradientMeasure
import TreePolicyGradient
import BatchReduction
import BatchPerformance
import FiniteSelection
import ProMaxCertificate
import EmpiricalCertificate
import GroupCertificate
import PolicyCover
import RLOOSurrogate
import SurrogateBridge
import ProMaxObjective
import PPODerivative
import AutoregressiveTree
import TreeSurrogate
import PromptMixture
import PrefixConcentration
import PrefixMasking
import StructuralExamples
import GroupFiltering
import ImplementationBridge
import PaddedPrefix
import NormalizationSafeguards
import AdaptiveMechanism
import LogitBounds
import UniformScale
import BinaryUpdate
import BinaryLength
import BinaryAscent
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
