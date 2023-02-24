export const fetchMLChatResponse = async (inputMessage: string) => {
  let response;
  const url = "/api/chat";
  console.log('url:', url)
  let output: string | null = null;

  try {
    response = await fetch(url, {
      headers: {
        'Content-Type': 'application/json'
      },
      method: "POST",
      body: JSON.stringify({ message: inputMessage }),
    });
  } catch (error) {
    console.error("Fetch Error:", error);
  }

  if (response?.ok) {
    const parsedRes = await response.json();
    if (
      parsedRes &&
      Object.hasOwn(parsedRes, "message") &&
      typeof parsedRes.message === "string"
    ) {
      output = parsedRes.message as string;
    }
  } else {
    console.error(`HTTP Response Code: ${response?.status}`);
  }
  return output;
};
