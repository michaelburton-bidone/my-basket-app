"use client";

import { useCart } from "@/contexts/ApiCartContext";
import type { CartItem, Order } from "@/lib/types";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Separator } from "@/components/ui/separator";
import { useToast } from "@/hooks/use-toast";
import { useRouter } from "next/navigation";
import Image from "next/image";
import Link from "next/link";
import { ShoppingBag, CheckCircle, Loader2 } from "lucide-react";
import { useState } from "react";
import { apiClient } from "@/lib/api/client";
import { getUserId } from "@/lib/session";

export function OrderReviewClient() {
  const { items, totalAmount: cartTotalAmount, clearCart, loading, error } = useCart();
  const { toast } = useToast();
  const router = useRouter();
  const [isOrderPlaced, setIsOrderPlaced] = useState(false);
  const [isPlacingOrder, setIsPlacingOrder] = useState(false);

  const handlePlaceOrder = async () => {
    if (!items || items.length === 0) {
      toast({
        title: "Cannot place order",
        description: "Your cart is empty.",
        variant: "destructive",
      });
      return;
    }

    try {
      setIsPlacingOrder(true);
      const userId = getUserId();
      
      // Create order through the order service
      const orderItems = items.map(item => ({
        id: item.id,
        name: item.name,
        price: item.price,
        quantity: item.quantity,
        description: item.description,
        image: item.image,
        dataAiHint: item.dataAiHint,
      }));
      
      await apiClient.createOrder(userId, orderItems);
      
      // Clear the cart after successful order
      await clearCart();
      
      setIsOrderPlaced(true);
      toast({
        title: "Order placed successfully!",
        description: "Your order has been placed and you will receive a confirmation email.",
      });
    } catch (error) {
      console.error('Error placing order:', error);
      toast({
        title: "Error placing order",
        description: "There was a problem placing your order. Please try again.",
        variant: "destructive",
      });
    } finally {
      setIsPlacingOrder(false);
    }
  };

  if (loading) {
    return (
      <div className="text-center py-12">
        <Loader2 className="mx-auto h-8 w-8 animate-spin text-muted-foreground mb-4" />
        <p className="text-muted-foreground">Loading your order...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="text-center py-12">
        <p className="text-destructive mb-4">Error loading order: {error}</p>
        <Button onClick={() => window.location.reload()}>Retry</Button>
      </div>
    );
  }

  if (isOrderPlaced) {
    return (
       <div className="container mx-auto py-12 px-4 sm:px-6 lg:px-8 flex flex-col items-center text-center" data-testid="order-placed-success">
        <CheckCircle className="h-24 w-24 text-green-500 mb-6" />
        <h1 className="text-3xl font-bold tracking-tight text-foreground sm:text-4xl mb-4" data-testid="order-success-message">
          Order Placed Successfully!
        </h1>
        <p className="text-lg text-muted-foreground mb-8">
          Thank you for your purchase. You can view your order details in your order history.
        </p>
        <Button asChild className="bg-primary text-primary-foreground hover:bg-primary/90" data-testid="view-orders-button">
          <Link href="/orders">View My Orders</Link>
        </Button>
      </div>
    )
  }
  

  if (!items || items.length === 0) {
    return (
      <div className="text-center py-12" data-testid="checkout-cart-empty">
        <ShoppingBag className="mx-auto h-16 w-16 text-muted-foreground mb-4" />
        <h2 className="text-2xl font-semibold text-foreground mb-2">Your cart is empty</h2>
        <p className="text-muted-foreground mb-6">Add items to your cart to proceed with checkout.</p>
        <Button asChild className="bg-accent text-accent-foreground hover:bg-accent/90">
          <Link href="/">Continue Shopping</Link>
        </Button>
      </div>
    );
  }

  return (
    <div className="max-w-2xl mx-auto" data-testid="order-review">
      <Card className="shadow-xl">
        <CardHeader>
          <CardTitle className="text-2xl">Review Your Order</CardTitle>
          <CardDescription>Please check your items and total before placing the order.</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-4 mb-6" data-testid="order-items">
            {items.map((item: CartItem) => (
              <div key={item.id} className="flex items-center space-x-4 p-3 border rounded-md" data-testid={`order-item-${item.id}`}>
                <div className="relative h-16 w-16 flex-shrink-0 overflow-hidden rounded-md border">
                  <Image
                    src={item.image}
                    alt={item.name}
                    fill
                    sizes="64px"
                    className="object-cover"
                    data-ai-hint={item.dataAiHint}
                    data-testid={`order-item-image-${item.id}`}
                  />
                </div>
                <div className="flex-1">
                  <h3 className="text-md font-medium text-foreground" data-testid={`order-item-name-${item.id}`}>{item.name}</h3>
                  <p className="text-sm text-muted-foreground" data-testid={`order-item-quantity-${item.id}`}>Quantity: {item.quantity}</p>
                </div>
                <p className="text-md font-semibold text-foreground" data-testid={`order-item-total-${item.id}`}>
                  ${(item.price * item.quantity).toFixed(2)}
                </p>
              </div>
            ))}
          </div>
          <Separator />
          <div className="mt-6 space-y-2">
            <div className="flex justify-between text-lg" data-testid="order-subtotal-row">
              <span className="text-muted-foreground">Subtotal</span>
              <span data-testid="order-subtotal-amount">${cartTotalAmount.toFixed(2)}</span>
            </div>
             <div className="flex justify-between text-lg" data-testid="order-shipping-row">
              <span className="text-muted-foreground">Shipping</span>
              <span>Free</span>
            </div>
            <Separator/>
            <div className="flex justify-between font-bold text-xl text-primary" data-testid="order-total-row">
              <span>Total</span>
              <span data-testid="order-total-amount">${cartTotalAmount.toFixed(2)}</span>
            </div>
          </div>
        </CardContent>
        <CardFooter>
          <Button 
            onClick={handlePlaceOrder} 
            className="w-full text-lg py-6 bg-accent text-accent-foreground hover:bg-accent/90"
            disabled={!items || items.length === 0 || isPlacingOrder}
            data-testid="place-order-button"
          >
            {isPlacingOrder ? (
              <>
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                Placing Order...
              </>
            ) : (
              "Place Order"
            )}
          </Button>
        </CardFooter>
      </Card>
    </div>
  );
}
