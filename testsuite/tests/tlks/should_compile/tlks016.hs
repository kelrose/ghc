{-# LANGUAGE TopLevelKindSignatures #-}
{-# LANGUAGE TypeFamilies, PolyKinds, ExplicitForAll #-}

module TLKS_016 where

import Data.Kind (Type)

type T :: forall k. k -> Type
data T (x :: j) :: Type
