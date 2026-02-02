"use client";

import Link from "next/link";
import { useCart } from "@/contexts/ApiCartContext";
import { CartItemCard } from "./CartItemCard";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Separator } from "@/components/ui/separator";
import { ShoppingBag, ArrowRight, Loader2 } from "lucide-react";
import { GrocerySuggestions } from "@/components/recommendations/GrocerySuggestions";

export function CartView() {
  const { items, totalAmount: cartTotalAmount, loading, error } = useCart();

  if (loading) {
    return (
      <div className="text-center py-12">
        <Loader2 className="mx-auto h-8 w-8 animate-spin text-muted-foreground mb-4" />
        <p className="text-muted-foreground">Loading your cart...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="text-center py-12">
        <p className="text-destructive mb-4">Error loading cart: {error}</p>
        <Button onClick={() => window.location.reload()}>Retry</Button>
      </div>
    );
  }

  if (items.length === 0) {
    return (
      <div className="text-center py-12" data-testid="cart-empty">
        <ShoppingBag className="mx-auto h-16 w-16 text-muted-foreground mb-4" />
        <h2 className="text-2xl font-semibold text-foreground mb-2" data-testid="empty-cart-message">Your cart is empty</h2>
        <p className="text-muted-foreground mb-6">Looks like you haven't added anything to your cart yet.</p>
        <Button asChild className="bg-accent text-accent-foreground hover:bg-accent/90" data-testid="continue-shopping-button">
          <Link href="/">Start Shopping</Link>
        </Button>
      </div>
    );
  }

  return (
    <div className="grid md:grid-cols-3 gap-8" data-testid="cart-view">
      <div className="md:col-span-2 space-y-4" data-testid="cart-items">
        {items.map((item) => (
          <CartItemCard key={item.id} item={item} />
        ))}
      </div>

      <div className="md:col-span-1 space-y-6">
        <Card className="shadow-lg" data-testid="order-summary">
          <CardHeader>
            <CardTitle className="text-xl">Order Summary</CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            <div className="flex justify-between" data-testid="subtotal-row">
              <span className="text-muted-foreground">Subtotal</span>
              <span data-testid="subtotal-amount">${cartTotalAmount.toFixed(2)}</span>
            </div>
            <div className="flex justify-between" data-testid="shipping-row">
              <span className="text-muted-foreground">Shipping</span>
              <span>Free</span>
            </div>
            <Separator />
            <div className="flex justify-between font-semibold text-lg" data-testid="total-row">
              <span>Total</span>
              <span className="text-primary" data-testid="total-amount">${cartTotalAmount.toFixed(2)}</span>
            </div>
          </CardContent>
          <CardFooter>
            <Button asChild className="w-full bg-primary text-primary-foreground hover:bg-primary/90" data-testid="checkout-button">
              <Link href="/checkout">
                Proceed to Checkout <ArrowRight className="ml-2 h-4 w-4" />
              </Link>
            </Button>
          </CardFooter>
        </Card>
        <GrocerySuggestions />
      </div>
    </div>
  );
}
