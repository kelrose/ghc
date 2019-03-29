{-# LANGUAGE TopLevelKindSignatures, ScopedTypeVariables #-}
{-# LANGUAGE PolyKinds, TypeApplications #-}

module TLKS_012 where

import Data.Kind (Type)

data T (a :: k)

type S :: forall k. k -> Type
type S = T @k
