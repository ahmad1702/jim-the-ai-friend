import { API_URL } from "../utils/constants";
import { getAPIUrl } from "../utils/utils";

export const fetchMLChatResponse = async (inputMessage: string) => {
  let response;
  const url = API_URL + "/chat";
  let output: string | null = null;

  try {
    response = await fetch(url, {
      headers: {
        'Content-Type': 'application/json'
        // 'Content-Type': 'application/x-www-form-urlencoded',
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
