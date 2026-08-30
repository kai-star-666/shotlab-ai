const DB_NAME = "shotlab-ai";
const STORE = "sessions";
const LIMIT = 30;

export function prepareSessionForPersistence(session) {
  return {
    ...session,
    shots: session.shots.map((shot) => {
      const { analysisFrames, curves, keyframes, ready, loading, release, ...analysis } = shot.analysis;
      return { ...shot, analysis: { ...analysis, charts: { angleCurves: [] }, skeletonVideo: { available: false, mode: "not-persisted", frameCount: 0 }, skeleton_video: { available: false, mode: "not-persisted" }, processedVideo: { available: false, mode: "not-persisted" }, processed_video: { available: false, mode: "not-persisted" } } };
    }),
  };
}

function requestResult(request) {
  return new Promise((resolve, reject) => { request.onsuccess = () => resolve(request.result); request.onerror = () => reject(request.error); });
}

export async function openSessionStore() {
  const request = indexedDB.open(DB_NAME, 1);
  request.onupgradeneeded = () => {
    const store = request.result.createObjectStore(STORE, { keyPath: "sessionId" });
    store.createIndex("updatedAt", "updatedAt");
  };
  const db = await requestResult(request);
  const transaction = (mode = "readonly") => db.transaction(STORE, mode).objectStore(STORE);
  return {
    async list() { return (await requestResult(transaction().getAll())).toSorted((a, b) => b.updatedAt.localeCompare(a.updatedAt)); },
    async get(sessionId) { return requestResult(transaction().get(sessionId)); },
    async save(session) {
      await requestResult(transaction("readwrite").put(prepareSessionForPersistence(session)));
      const all = await this.list();
      for (const stale of all.slice(LIMIT)) await requestResult(transaction("readwrite").delete(stale.sessionId));
      return session;
    },
    async delete(sessionId) { await requestResult(transaction("readwrite").delete(sessionId)); },
    async clear() { await requestResult(transaction("readwrite").clear()); },
    async latestActive() { return (await this.list()).find((session) => session.status === "active") || null; },
    close() { db.close(); },
  };
}
