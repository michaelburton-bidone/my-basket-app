"use client";

import Image from "next/image";
import type { CartItem } from "@/lib/types";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Trash2 } from "lucide-react";
import { useCart } from "@/contexts/ApiCartContext";
import { useToast } from "@/hooks/use-toast";
import { useState } from "react";

interface CartItemCardProps {
  item: CartItem;
}

export function CartItemCard({ item }: CartItemCardProps) {
  const { removeFromCart } = useCart();
  const { toast } = useToast();
  const [isRemoving, setIsRemoving] = useState(false);

  const handleRemoveItem = async () => {
    try {
      setIsRemoving(true);
      await removeFromCart(item.id);
      toast({
        title: "Item removed",
        description: `${item.name} has been removed from your cart.`,
        variant: "destructive",
      });
    } catch (error) {
      toast({
        title: "Error",
        description: "Failed to remove item from cart.",
        variant: "destructive",
      });
    } finally {
      setIsRemoving(false);
    }
  };

  return (
    <Card className="flex items-center p-4 space-x-4 shadow-sm" data-testid={`cart-item-${item.id}`}>
      <div className="relative h-20 w-20 flex-shrink-0 overflow-hidden rounded-md border border-gray-200">
        <Image
          src={item.image}
          alt={item.name}
          fill
          sizes="80px"
          className="object-cover"
          data-ai-hint={item.dataAiHint}
          data-testid={`cart-item-image-${item.id}`}
        />
      </div>
      <CardContent className="flex flex-1 flex-col justify-between p-0">
        <div>
          <h3 className="text-md font-medium text-foreground" data-testid={`cart-item-name-${item.id}`}>{item.name}</h3>
          <p className="text-sm text-muted-foreground" data-testid={`cart-item-quantity-${item.id}`}>
            Quantity: {item.quantity}
          </p>
        </div>
        <div className="flex items-center justify-between mt-2">
          <p className="text-md font-semibold text-primary" data-testid={`cart-item-price-${item.id}`}>
            ${(item.price * item.quantity).toFixed(2)}
          </p>
          <Button
            variant="ghost"
            size="icon"
            onClick={handleRemoveItem}
            disabled={isRemoving}
            className="text-destructive hover:bg-destructive/10"
            data-testid={`remove-item-${item.id}`}
          >
            <Trash2 className="h-5 w-5" />
            <span className="sr-only">Remove item</span>
          </Button>
        </div>
      </CardContent>
    </Card>
  );
}
