import Link from "next/link";

export default function NotFound() {
  return (
    <main>
      <h1>404</h1>
      <p>페이지를 찾을 수 없습니다.</p>
      <Link href="/">메인으로 돌아가기</Link>
    </main>
  );
}
