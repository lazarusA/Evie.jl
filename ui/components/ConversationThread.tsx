"use client";

import { MessageCircleDashedIcon } from "lucide-react";

import { Avatar, AvatarFallback } from "@/components/ui/avatar";
import { Bubble, BubbleContent } from "@/components/ui/bubble";
import { Marker, MarkerContent } from "@/components/ui/marker";
import {
  Message,
  MessageAvatar,
  MessageContent,
  MessageFooter,
} from "@/components/ui/message";
import {
  MessageScroller,
  MessageScrollerButton,
  MessageScrollerContent,
  MessageScrollerProvider,
  MessageScrollerViewport,
} from "@/components/ui/message-scroller";

// ── Types ─────────────────────────────────────────────────────────────────────

export interface MessageData {
  id: string;
  role: "user" | "evie";
  text: string;
  /** true while the LLM is still streaming this message */
  streaming?: boolean;
}

interface ConversationThreadProps {
  messages: MessageData[];
  /** Show the STT "transcribing your voice" indicator */
  transcribing?: boolean;
}

// ── Single message ────────────────────────────────────────────────────────────

function ChatMessage({ message }: { message: MessageData }) {
  const isUser = message.role === "user";

  return (
    <Message align={isUser ? "end" : "start"}>
      <MessageAvatar>
        <Avatar className="h-7 w-7">
          <AvatarFallback>{isUser ? "U" : "E"}</AvatarFallback>
        </Avatar>
      </MessageAvatar>
      <MessageContent>
        <Bubble variant={isUser ? undefined : "muted"}>
          <BubbleContent>{message.text}</BubbleContent>
        </Bubble>
        {message.streaming && (
          <MessageFooter>Evie is typing…</MessageFooter>
        )}
      </MessageContent>
    </Message>
  );
}

// ── Thread ────────────────────────────────────────────────────────────────────

export function ConversationThread({
  messages,
  transcribing = false,
}: ConversationThreadProps) {
  if (messages.length === 0 && !transcribing) {
    return (
      <div className="flex h-full flex-col items-center justify-center gap-2 text-center select-none py-16">
        <MessageCircleDashedIcon className="size-8 text-muted-foreground" strokeWidth={1.5} />
        <p className="text-sm text-muted-foreground">
          Tap the mic and start talking.
          <br />
          Evie is listening.
        </p>
      </div>
    );
  }

  return (
    <MessageScrollerProvider>
      <MessageScroller className="h-full">
        <MessageScrollerViewport>
          <MessageScrollerContent
            aria-busy={transcribing}
            className="flex flex-col gap-6 px-1 py-2"
          >
            {messages.map((msg) => (
              <ChatMessage key={msg.id} message={msg} />
            ))}

            {/* STT in-progress indicator */}
            {transcribing && (
              <Marker role="status" className="self-end">
                <MarkerContent className="shimmer">
                  Listening…
                </MarkerContent>
              </Marker>
            )}
          </MessageScrollerContent>
        </MessageScrollerViewport>
        <MessageScrollerButton />
      </MessageScroller>
    </MessageScrollerProvider>
  );
}
