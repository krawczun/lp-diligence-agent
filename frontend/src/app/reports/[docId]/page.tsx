import { ChecklistRunner } from "./checklist-runner";
import { fetchDocuments, fetchChecklistItems } from "@/lib/api";

export const dynamic = "force-dynamic";

export default async function ReportPage({ params }: { params: Promise<{ docId: string }> }) {
  const { docId } = await params;
  const [docs, items] = await Promise.all([fetchDocuments(), fetchChecklistItems()]);
  const doc = docs.find((d) => d.doc_id === docId);

  if (!doc) {
    return (
      <div className="text-neutral-400">
        Unknown document: <span className="mono">{docId}</span>
      </div>
    );
  }

  return (
    <div className="space-y-8">
      <header className="space-y-1">
        <div className="text-sm text-neutral-500 mono">{doc.doc_id}</div>
        <h1 className="text-2xl font-semibold">
          {doc.entity} · {doc.period}
        </h1>
        <div className="text-sm text-neutral-400">{doc.doc_type}</div>
      </header>

      <ChecklistRunner docId={doc.doc_id} items={items} />
    </div>
  );
}
