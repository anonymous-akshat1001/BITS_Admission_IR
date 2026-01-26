'use client';

import React, { useState, ChangeEvent, useEffect, useRef } from 'react';
import { Button } from "@/components/ui/button";
import { LuSend, LuExternalLink } from "react-icons/lu";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import Markdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import Loader from '@/components/shared/Loader';
import HomeStarter from '@/components/shared/HomeStarter';
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger
} from "@/components/ui/accordion";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL;

if (!API_BASE_URL) {
  console.warn("NEXT_PUBLIC_API_URL is not defined");
}

interface SourceDocumentMetadata {
  file_path?: string;
  doc_name?: string;
  source?: string;
  score?: number;
  [key: string]: any;
}

interface SourceDocument {
  page_content: string;
  metadata: SourceDocumentMetadata;
}

interface Message {
  type: 'user' | 'bot';
  text: string;
  source_documents?: SourceDocument[];
}

export default function Home() {
  const [userQuery, setUserQuery] = useState("");
  const [messages, setMessages] = useState<Message[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [useReranker] = useState(false);
  const chatContainerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    chatContainerRef.current?.scrollTo({
      top: chatContainerRef.current.scrollHeight,
      behavior: "smooth",
    });
  }, [messages]);

  const sendMessage = async (message: string) => {
    if (!message.trim() || !API_BASE_URL) return;

    setIsLoading(true);
    setMessages(prev => [...prev, { type: 'user', text: message }]);
    setUserQuery("");

    const endpoint = useReranker
      ? `${API_BASE_URL}/query/hybrid-rerank/`
      : `${API_BASE_URL}/query/hybrid/`;

    try {
      const res = await fetch(endpoint, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: message }),
      });

      if (!res.ok) throw new Error(res.statusText);

      const data = await res.json();

      setMessages(prev => [
        ...prev,
        {
          type: 'bot',
          text: data.answer,
          source_documents: data.source_documents,
        },
      ]);
    } catch (err) {
      console.error(err);
      setMessages(prev => [
        ...prev,
        {
          type: 'bot',
          text: "Sorry, I couldn't connect to the backend. Please try again in a moment.",
        },
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  const getDocName = (meta: SourceDocumentMetadata) =>
    meta.doc_name || meta.file_path?.split('/').pop() || "Source Document";

  return (
    <>
      <main className="flex min-h-screen flex-col items-center px-4 py-24 pb-40 bg-gray-50">
        <section
          ref={chatContainerRef}
          className="w-full max-w-3xl flex flex-col gap-4 overflow-y-auto"
        >
          {messages.length === 0 && <HomeStarter />}

          {messages.map((m, i) => (
            <div key={i} className={`flex gap-3 ${m.type === 'user' ? 'justify-end' : ''}`}>
              {m.type === 'bot' && (
                <Avatar>
                  <AvatarImage src="/user2.png" />
                  <AvatarFallback>BOT</AvatarFallback>
                </Avatar>
              )}

              <div className={`rounded-xl px-4 py-3 max-w-[80%] text-sm bg-white shadow`}>
                <Markdown remarkPlugins={[remarkGfm]}>{m.text}</Markdown>

                {m.type === 'bot' &&
                  m.source_documents &&
                  m.source_documents.length > 0 && (
                  <Accordion type="single" collapsible className="mt-3">
                    {m.source_documents.map((doc, idx) => (
                      <AccordionItem key={idx} value={`doc-${idx}`}>
                        <AccordionTrigger className="text-xs">
                          {getDocName(doc.metadata)}
                        </AccordionTrigger>
                        <AccordionContent className="text-xs">
                          <p className="mb-2 whitespace-pre-wrap">{doc.page_content}</p>
                          {doc.metadata.source && (
                            <a
                              href={doc.metadata.source}
                              target="_blank"
                              rel="noopener noreferrer"
                              className="inline-flex items-center gap-1 text-orange-600"
                            >
                              Open source <LuExternalLink />
                            </a>
                          )}
                        </AccordionContent>
                      </AccordionItem>
                    ))}
                  </Accordion>
                )}
              </div>

              {m.type === 'user' && (
                <Avatar>
                  <AvatarImage src="/useres.png" />
                  <AvatarFallback>YOU</AvatarFallback>
                </Avatar>
              )}
            </div>
          ))}

          {isLoading && <Loader />}
        </section>
      </main>

      <footer className="fixed bottom-0 w-full bg-white border-t p-3">
        <div className="max-w-3xl mx-auto flex gap-2">
          <input
            value={userQuery}
            onChange={(e) => setUserQuery(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && sendMessage(userQuery)}
            disabled={isLoading}
            className="flex-1 border rounded-xl px-4 py-2 text-sm outline-none"
            placeholder="Ask something..."
          />
          <Button
            onClick={() => sendMessage(userQuery)}
            disabled={isLoading || !userQuery.trim()}
            className="rounded-xl bg-orange-500 hover:bg-orange-600"
          >
            <LuSend className="text-white" />
          </Button>
        </div>
        <p className="text-[11px] text-center mt-1 text-gray-500">
          Disclaimer: AI-generated responses may contain errors.
        </p>
      </footer>
    </>
  );
}