{-# LANGUAGE TopLevelKindSignatures, ScopedTypeVariables #-}
{-# LANGUAGE PolyKinds #-}

module TLKS_011 where

import Data.Kind (Type)
import Data.Proxy (Proxy)

type T :: forall k. k -> Type
data T a = MkT (Proxy (a :: k))
