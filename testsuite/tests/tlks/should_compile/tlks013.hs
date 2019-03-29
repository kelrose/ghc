{-# LANGUAGE TopLevelKindSignatures, ScopedTypeVariables #-}
{-# LANGUAGE PolyKinds, ConstraintKinds #-}

module TLKS_013 where

import Data.Kind (Constraint)

data T (a :: k)

type C :: forall k. k -> Constraint
class C a where
  getC :: T (a :: k)
